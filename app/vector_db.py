import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import uuid


class ChromaNativeClient:
    def __init__(self, host: str='localhost',
                 port: int=8000,
                 collection_name: str=None):
        self.client = chromadb.HttpClient(host=host,
                                          port=port)
        self.collection = self.client.create_collection(collection_name)

    def load_documents(self, tokenized_text):
        for doc in tokenized_text:
            self.collection.add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content,
                embeddings=doc.embeddings
            )


class ChromaLangChainClient:
    def __init__(self, host: str='localhost',
                 port: int=8000,
                 collection_name: str=None):
        self.embedding_function = OpenAIEmbeddings()
        # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.client = chromadb.HttpClient(host=host,
                                          port=port)
        self.collection_name = collection_name
        self.vdb = None

    def initialize_vdb(self):
        self.vdb = Chroma(client=self.client,
                          collection_name=self.collection_name,
                          embedding_function=self.embedding_function,
                         )


