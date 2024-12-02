from pymilvus import MilvusClient


client = MilvusClient("milvus_database.db")

client.create_collection(
    collection_name="my_collection",
    dimension=768
)
