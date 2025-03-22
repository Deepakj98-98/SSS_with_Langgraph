from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

class Qdrant_retrieval:
    def __init__(self,collection_name):
        self.embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        load_dotenv()
        print("here")
        print("Initializing Qdrant client...")
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

        

