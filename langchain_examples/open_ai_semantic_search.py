import getpass
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Example of creating a document
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Example document loader
file_path = "nke-10k-2023.pdf" #replace with your doc path
loader = PyPDFLoader(file_path)
docs = loader.load()

print ("Document length")
print (len(docs))

print ("Further info on the doc")
print (f"{docs[0].page_content[:200]}\n")
print (docs[0].metadata)

# Example Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print ("The length of all the splits")
print (len(all_splits))

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
print ("Embeddings examples")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print (f"Generated vectors of length {len(vector_1)}\n")
print (vector_1[:10])

print ("Example of an in-memory vector store")

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)
print (ids)

print ("Return results based on similarity to a string query")
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print (results[0])

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print (f"Score: {score}\n")
print (doc)

print ("Return documents based on similarity to an embedded query")

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print (results[0])
