#!/usr/bin/env python3
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask import render_template
from flask import request
import math
import json
from azure.cosmos import CosmosClient
import redis
from sklearn.neighbors import NearestNeighbors
import time

redis_passwd = "dH83jFtaYDZA7socl4hld3cGCgMu81TCrAzCaMSKXMs="
redis_host = "LiushuaiRedis.redis.cache.windows.net"
cache = redis.StrictRedis(
    host=redis_host, port=6380,
    db=0, password=redis_passwd,
    ssl=True,
)

if cache.ping():
    print("pong")


DB_CONN_STR ="AccountEndpoint=https://tutorial-uta-cse6332.documents.azure.com:443/;" \
             "AccountKey=fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw==;"
db_client = CosmosClient.from_connection_string(conn_str=DB_CONN_STR)
database = db_client.get_database_client("tutorial")

app = Flask(__name__)

@app.route('/purge_cache', methods=['GET'])
def purge_cache():
    for key in cache.keys():
        cache.delete(key.decode())
    return "Cache cleared successfully"
def calculate_distance(lat1, lng1, lat2, lng2):

    # 直线距离计算方法
    # distance = ((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2) ** 0.5

    # 球面距离计算方法
    # 将经纬度转换为弧度
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)

    # 使用 Haversine 公式计算球面距离
    delta_lat = lat2_rad - lat1_rad
    delta_lng = lng2_rad - lng1_rad
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371 * c  # 地球平均半径为 6371 公里

    return distance


@app.route('/stat/closest_cities', methods=['GET'])
def get_close_cities():
    container = database.get_container_client("us_cities")

    city_name = request.args.get('city')
    page_size = int(request.args.get('page_size'))
    page = int(request.args.get('page'))

    # 尝试从缓存中获取结果
    redis_key = f"{city_name}_{page_size}_{page}"
    cached_result = cache.get(redis_key)

    if cached_result is not None:
        cached_result = json.loads(cached_result)
        if isinstance(cached_result, dict):
            response = {'redis_key': 1}
            response.update(cached_result)
            return jsonify(response)

    city_data = list(container.query_items(
        query="SELECT c.lat, c.lng FROM c WHERE c.city = @city",
        parameters=[dict(name="@city", value=city_name)],
        enable_cross_partition_query=True
    ))

    if len(city_data) == 0:
        return jsonify({'error': 'City not found'})

    city_info = city_data[0]
    lat1 = float(city_info['lat'])
    lng1 = float(city_info['lng'])

    # 从数据库中获取所有城市的经纬度，并计算距离
    all_cities = container.query_items(
        query="SELECT c.city, c.lat, c.lng FROM c",
        enable_cross_partition_query=True
    )

    cities_with_distance = []
    for city in all_cities:
        lat2 = float(city['lat'])
        lng2 = float(city['lng'])
        distance = calculate_distance(lat1, lng1, lat2, lng2)
        cities_with_distance.append({
            'city': city['city'],
            'distance': distance
        })

    # 按照距离升序排序
    cities_with_distance.sort(key=lambda c: c['distance'])

    # 分页返回结果
    start_index = page * page_size
    end_index = start_index + page_size
    closest_cities = cities_with_distance[start_index:end_index]

    start_time = time.time()

    # 构建响应结果
    response = {
        'closest_cities': closest_cities,
        'response_time': time.time() - start_time,
    }

    # 将结果存入缓存
    # 将字典对象转换为 JSON 格式的字符串
    encoded_result = json.dumps(response)
    # 存入 Redis
    cache.set(redis_key, encoded_result)

    # 返回结果
    return jsonify(response)

@app.route('/stat/knn_reviews', methods=['GET'])
def get_knn_words():
    # 获取请求参数
    classes = int(request.args.get('classes'))
    k = int(request.args.get('k'))
    words_num = int(request.args.get('words'))

    # 尝试从缓存中获取结果
    redis_key = f"{classes}_{k}_{words_num}"
    cached_result = cache.get(redis_key)

    if cached_result is not None:
        print(redis_key)
        cached_result = json.loads(cached_result)
        if isinstance(cached_result, list):
            response = {'redis_key': 1}
            cached_result_with_response = [dict(d, **response) for d in cached_result]
            return jsonify(cached_result_with_response)

    # 数据库连接
    container = database.get_container_client("us_cities")

    # 查询城市数据
    query = "SELECT c.id, c.city, c.lat, c.lng, c.population FROM c"
    cities = container.query_items(query, enable_cross_partition_query=True)

    # 构建城市列表和欧拉距离矩阵
    city_list = []
    distance_matrix = []
    for city in cities:
        city_list.append(city)
        distances = []
        for other_city in city_list:
            distance = calculate_distance(float(city['lat']), float(city['lng']), float(other_city['lat']),
                                          float(other_city['lng']))
            distances.append(distance)
        distance_matrix.append(distances)

    # KNN算法聚类
    clusters = [[] for _ in range(classes)]
    for i in range(len(city_list)):
        city_distances = distance_matrix[i]
        nearest_neighbors = sorted(range(len(city_distances)), key=lambda x: city_distances[x])[:k]
        cluster_index = i % classes
        clusters[cluster_index].append(city_list[i])

    # 加载停用词
    stopwords = set()
    with open('stopwords.txt', 'r') as file:
        for line in file:
            word = line.strip()
            stopwords.add(word)

    print("success1")
    # 数据库
    container_reviews = database.get_container_client("reviews")
    query_review = "SELECT c.score,c.city,c.review FROM c"
    reviews = list(container_reviews.query_items(query_review, enable_cross_partition_query=True))

    print("success2")

    # 在每个类别中进行进一步处理
    response = []
    for i in range(classes):
        cluster = clusters[i]
        center_city = cluster[0]  # 假设第一个城市为中心城市

        # 获取该类别中所有城市的评论和人口
        comments = []
        total_population = 0
        for city in cluster:
            city_comments = [
                {
                    'review': review['review'],
                    'score': review['score']
                }
                for review in reviews if review['city'] == city['city']
            ]
            comments.extend(city_comments)

            # 获取城市的人口
            population = city['population']
            total_population += float(population)
        print("success3")
        # 统计单词频次
        word_counts = {}
        for comment in comments:
            comment_text = comment['review']
            words = comment_text.split()
            for word in words:
                word = word.lower()
                if word not in stopwords:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # 获取前n个最受欢迎的单词
        popular_words = sorted(word_counts, key=word_counts.get, reverse=True)[:words_num]

        # print(comments)
        # 计算评论得分总和
        total_score = 0
        for comment in comments:
            score = int(comment['score'])
            total_score += score

        # 计算每千人口的平均得分
        average_score_per_thousand = (total_score / total_population) * 1000

        # 计算代码执行时间
        start_time = time.time()

        # 构建类别结果
        result = {
            'class_number': i + 1,
            'center_city': center_city['city'],
            'cities': [city['city'] for city in cluster],
            'popular_words': popular_words,
            'average_score_per_thousand': average_score_per_thousand,
            'response_time': float(time.time() - start_time)
        }
        # print(result)
        response.append(result)

    # 将结果存入缓存
    # 将字典对象转换为 JSON 格式的字符串
    encoded_results = json.dumps(response)
    # 存入 Redis
    cache.set(redis_key, encoded_results)

    # 返回结果给前端
    return jsonify(response)


@app.route("/", methods=['GET'])
def index():
    message = "Congratulations, it's a web app!"
    return render_template(
        'index.html',
        message=message,
    )



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)