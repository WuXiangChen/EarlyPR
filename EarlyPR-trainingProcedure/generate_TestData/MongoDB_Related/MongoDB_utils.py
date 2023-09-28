
'''导包区'''
from pymongo import MongoClient
from generate_TestData.MongoDB_Related.CONSTANT import mongodb_url,database_name
from pymongo.errors import CollectionInvalid
import json
# 数据库连接
# 连接到MongoDB

'''连接MongoDB'''
def mongo_conn():
    client = MongoClient(mongodb_url)
    database = client[database_name]
    return database

def prepare_query_work(database,collection_name, query):
    try:
        collection = database[collection_name]
        result = collection.aggregate(query,allowDiskUse = True)
        return list(result)
    except CollectionInvalid as ci:
        print(f"Error accessing collection: {ci}")
        return None
    except Exception as e:
        print(collection_name)
        print(query)
        print(f"An error occurred during query preparation: {e}")
        print()
        return None
import redis

class RedisStorage:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)
    def set_key_value(self, key, value):
        """
        存储键值对到Redis数据库
        """
        return self.redis_client.set(key, value)

    def get_value(self, key):
        """
        获取指定键的值
        """
        return self.redis_client.get(key)

    def delete_key(self, key):
        """
        删除指定的键
        """
        return self.redis_client.delete(key)

    def exists_key(self, key):
        """
        检查键是否存在
        """
        return self.redis_client.exists(key)

    def append_value(self, key, value):
        """
        将值追加到已有键的值中
        """
        old_values = self.get_value(key)
        old_values_dict = {}
        if old_values!=None and json.loads(old_values)!=None and isinstance(json.loads(old_values),dict):
            old_values_dict = json.loads(old_values)
            old_values_dict.update(value)
        else:
            old_values_dict.update(value)

        new_values = json.dumps(old_values_dict)
        return self.redis_client.set(key, new_values)

database = mongo_conn()

#rs = RedisStorage()
