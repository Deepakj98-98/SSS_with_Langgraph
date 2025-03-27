from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class Qdrant_retrieval:
    def __init__(self,collection_name):
        self.embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        load_dotenv()
        print("here")
        print("Initializing Qdrant client...")
        # Check if the collection exists
        # Load Qdrant client
        self.qdrant_client = QdrantClient(url=os.getenv("ENV_URL"), api_key=os.getenv("API_KEY"))
        collections = self.qdrant_client.get_collections()
        existing_collections = [c.name for c in collections.collections]

        if collection_name not in existing_collections:
            print(f"Collection '{collection_name}' does not exist. Creating it...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Ensure the size matches your embeddings
            )
        self.qdrant=QdrantVectorStore.from_existing_collection(
            embedding=self.embedding_model,
            url=os.getenv("ENV_URL"),
            api_key=os.getenv("API_KEY"),
            collection_name=collection_name
        )

    def qdrant_retrieve(self, query):
        print("here")
        results = self.qdrant.similarity_search(query, k=2)
        retrieve_chunks=[]
        for i, doc in enumerate(results,1):
            retrieve_chunks.append(doc.page_content)
        return retrieve_chunks
    
'''
if __name__=="__main__":
    qc=Qdrant_retrieval("test_check")
    chunks=qc.qdrant_retrieve("What is a database")
    print(chunks)
'''

        

