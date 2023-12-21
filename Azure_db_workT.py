import uuid
from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient, PartitionKey, database
import os
import numpy as np
from sklearn.cluster import KMeans
import math
import time
def fetch_data(city_name=None, include_header=False, exact_match=False):

    DB_CONN_STR = "AccountEndpoint=https://tutorial-uta-cse6332.documents.azure.com:443/;AccountKey=fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw==;"


    db_client = CosmosClient.from_connection_string(conn_str=DB_CONN_STR)
    database = db_client.get_database_client("tutorial")

    container_cities = database.get_container_client("us_cities")
    container_reviews = database.get_container_client("reviews")
    print(container_cities)
    print(container_reviews)

    QUERY = "SELECT * from reviews"
    params = None
    if city_name is not None:
        QUERY = "SELECT * FROM us_cities p WHERE p.city = @city_name"
        params = [dict(name="@city_name", value=city_name)]
        if not exact_match:
            QUERY = "SELECT * FROM us_cities p WHERE p.city like @city_name"

    headers = ["city", "lat", "lng", "country", "state", "population"]
    result = []
    row_id = 0
    if include_header:
        line = [x for x in headers]
        line.insert(0, "")
        result.append(line)

    for item in container_cities.query_items(
            query=QUERY, parameters=params, enable_cross_partition_query=True,
    ):
        row_id += 1
        line = [str(row_id)]
        for col in headers:
            line.append(item[col])
        result.append(line)
    #print(result)

    return result
def fetch_reviews(city_name=None, include_header=False, exact_match=False):

    DB_CONN_STR = "AccountEndpoint=https://tutorial-uta-cse6332.documents.azure.com:443/;AccountKey=fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw==;"
    db_client = CosmosClient.from_connection_string(conn_str=DB_CONN_STR)
    database = db_client.get_database_client("tutorial")

    container_cities = database.get_container_client("us_cities")
    container_reviews = database.get_container_client("reviews")
    print(container_cities)
    print(container_reviews)

    QUERY = "SELECT * from reviews"
    params = None
    if city_name is not None:
        QUERY = "SELECT * FROM reviews p WHERE p.city = @city_name"
        params = [dict(name="@city_name", value=city_name)]
        if not exact_match:
            QUERY = "SELECT * FROM reviews p WHERE p.city like @city_name"

    headers = ["score", "city", "title", "review"]
    result = []
    row_id = 0
    if include_header:
        line = [x for x in headers]
        line.insert(0, "")
        result.append(line)

    for item in container_cities.query_items(
            query=QUERY, parameters=params, enable_cross_partition_query=True,
    ):
        row_id += 1
        line = [str(row_id)]
        for col in headers:
            line.append(item[col])
        result.append(line)
    #print(result)

    return result


def insert_data():

    DB_CONN_STR = "AccountEndpoint=https://liushuaisql.documents.azure.com:443/;" \
                  "AccountKey=oNej1J9istoCZngAmEvCgBZaLGHu9no9Nt09kzx6qjAsWg9nVgBWNa4QEAo2EKNqxUwBnqs99ifGACDbkfZQrw=="
    db_client = CosmosClient.from_connection_string(conn_str=DB_CONN_STR)
    database = db_client.get_database_client("us-cities")

    container = database.get_container_client("us-cities")
    # print(container)
    container.upsert_item({
        'city': 'test',
        'lat': '23.65',
        'lng': '-86',
        'country':'test',
        'state':'test',
        'population':'2000',
        'id':'22'
    }
    )
    return 'success insert'

def delete_data(city_name=None, include_header=False, exact_match=False):

    DB_CONN_STR = "AccountEndpoint=https://liushuaisql.documents.azure.com:443/;" \
                  "AccountKey=oNej1J9istoCZngAmEvCgBZaLGHu9no9Nt09kzx6qjAsWg9nVgBWNa4QEAo2EKNqxUwBnqs99ifGACDbkfZQrw=="
    db_client = CosmosClient.from_connection_string(conn_str=DB_CONN_STR)
    database = db_client.get_database_client("us-cities")

    container = database.get_container_client("us-cities")
    if city_name is not None:
        QUERY = "SELECT * FROM us_cities p WHERE p.city = @city_name"
        params = [dict(name="@city_name", value=city_name)]
        if not exact_match:
            QUERY = "SELECT * FROM us_cities p WHERE p.city like @city_name"
    for item in container.query_items(
            query=QUERY,
            enable_cross_partition_query=True):
        container.delete_item(item, partition_key='')


    return 'success delete'

if __name__ == '__main__':
    cn='Parma'
    #start_time = time.time()
    #curCity=fetch_data(city_name = cn)

    # distance=[]
    # for cur in curCity:
    #     for c in allCities:
    #         if c[1] !=cn:
    #             dis=math.sqrt((float(c[2]) - float(cur[2]))**2 + (float(c[3]) - float(cur[3]))**2)
    #             citydis={'queryCity':cn,'city':c[1],'distance':dis}
    #             distance.append(citydis)
    #
    # end_time = time.time()

    #insert_data()
    #delete_data(city_name = "test")
    cities = fetch_data(city_name=None)
    # 计算城市之间的距离矩阵
    num_cities = len(cities)
    print(len)
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            city_i = cities[i]
            city_j = cities[j]
            distance = np.sqrt((float(city_i[2] )- float(city_j[2])) ** 2 + (float(city_i[3])- float(city_j[3])) ** 2)
            distance_matrix[i, j] = distance

    # 输出距离矩阵
    # for row in distance_matrix:
    #     print(row)
    # KNN 聚类
    k = 2  # 聚类的数量
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(distance_matrix)

    # 输出聚类结果
    for i in range(k):
        cities_in_cluster = np.where(clusters == i)[0]
        print("Cluster", i + 1, ": Cities", cities_in_cluster)
    # 获取每个类别的中心城市及其他城市
    cluster_centers = kmeans.cluster_centers_
    AllReviews=[];
    for i in range(k):
        cluster_indices = np.where(clusters == i)[0]
        center_index = np.argmin(cluster_centers[i])  # 中心城市的索引
        reviews=[]
        center_city = cities[center_index]  # 中心城市对象
        reviews.append(fetch_reviews(center_city))
        other_cities = [cities[j] for j in cluster_indices if j != center_index]  # 其他城市对象
        print("Cluster", i + 1, ": Center City:", center_city.name)
        print("Other Cities:")
        for city in other_cities:
            print(city.name)
        print()

