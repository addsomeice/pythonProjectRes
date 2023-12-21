import redis
# password is the "Primary" copied in "Access keys"
redis_passwd = "dH83jFtaYDZA7socl4hld3cGCgMu81TCrAzCaMSKXMs="
# "Host name" in properties
redis_host = "LiushuaiRedis.redis.cache.windows.net"
# SSL Port
redis_port = 6380

cache = redis.StrictRedis(
            host=redis_host, port=redis_port,
            db=0, password=redis_passwd,
            ssl=True,
        )

if cache.ping():
    print("pong")
