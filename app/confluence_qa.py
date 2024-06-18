from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import re
import nltk.corpus
import warnings
from dotenv import load_dotenv
nltk.download('stopwords')
warnings.filterwarnings('ignore')


class ConfluenceQA:
	def __init__(self,
				 config: dict = {}):
		self.config = config
		self.embedding = self.init_embeddings()
		self.llm_name = self.init_model()
		self.loader = ConfluenceLoader(url=config['url'],
									   username=config['user'],
									   api_key=config['api_key'])

		self.vectordb = None
		self.texts = None
		self.retroever = None
		self.qa = None

	def init_embeddings(self) -> None:
		if 'embedding' in [i.lower() for i in self.config.keys()]:
			return self.config['embedding']
		else:
			return OpenAIEmbeddings()

	def init_model(self) -> None:

		if 'llm_name' in [i.lower() for i in self.config.keys()]:
			return ChatOpenAI(model_name=self.config['llm_name'],
							  temperature=0)
		else:
			return ChatOpenAI(model_name="gpt-3.5-turbo",
							  temperature=0)

	def get_chunk_documents(self) -> None:
		documents = self.loader.load(space_key=self.config['space_key'],
									 limit=100)

		text_splitter = CharacterTextSplitter(chunk_size=100,
											  chunk_overlap=0)
		texts = text_splitter.split_documents(documents)
		text_splitter = TokenTextSplitter(chunk_size=1000,
										  chunk_overlap=10,
										  encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
		self.texts = text_splitter.split_documents(texts)

	def vector_db_confluence_docs(self, force_reload: bool = False) -> None:
		self.vectordb = Chroma.from_documents(documents=self.texts,
											  embedding=self.embedding)

	def retreival_qa_chain(self):
		self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})
		self.qa = RetrievalQA.from_chain_type(llm=self.llm,
											  chain_type="stuff",
											  retriever=self.retriever)

	def answer_confluence(self, question: str) -> str:
		answer = self.qa.run(question)
		return answer

	def run_qa_setup(self) -> None:
		self.get_chunk_documents()
		self.vector_db_confluence_docs()
		self.qa()

	@staticmethod
	def clean_text(text: str) -> str:
		text = text.lower()
		text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
		stop = stopwords.words('english')
		text = " ".join([word for word in text.split() if word not in (stop)])
		text = re.sub("(\n+)", "", text)
		text = re.sub("(\s){2,}", " ", text)
		return text



