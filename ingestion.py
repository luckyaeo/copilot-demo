from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://www.investopedia.com/terms/c/convertible-bond-arbitrage.asp",
    "https://www.wallstreetmojo.com/convertible-arbitrage/",
    "https://www.stockgro.club/learn/share-market/convertible-arbitrage/",
    "https://analystprep.com/study-notes/cfa-level-2/relative-value-strategies-fixed-income-arbitrage/"
]

# Load the document from the web
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

# Split the document into sentences
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=20)
doc_split = splitter.split_documents(doc_list)

# Retriever
retriever = Chroma(
    collection_name="converts-rag",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings()
).as_retriever()

