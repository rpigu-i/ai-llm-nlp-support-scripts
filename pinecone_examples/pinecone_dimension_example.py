import getpass
import os
from pinecone import Pinecone, ServerlessSpec

if not os.environ.get("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter API key for Pinecone: ")

pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "pinecone-dimension-py"
dimension = 3072

spec = ServerlessSpec(cloud="aws", region="us-east-1")

pinecone.create_index(
    name=index_name,
    dimension=dimension,
    spec=spec
)
