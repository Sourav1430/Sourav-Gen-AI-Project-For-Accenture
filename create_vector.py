import os
import time
import shutil
import chromadb
from dotenv import load_dotenv
load_dotenv()

#clear DB Directory
def clear():
    shutil.rmtree('DB')
    time.sleep(2)
    os.mkdir('DB')
    print("#clear DB Directory")
clear()

#Create PersistentClient local DB
client = chromadb.PersistentClient(path="./DB")
collection = client.get_or_create_collection(name="Ascension_DB")
print("#Create PersistentClient local DB")

#PDF Loader
from langchain_community.document_loaders import PyPDFLoader
file_path = r'C:\Sourav\Virtual_Env\Ascension_AI_Project\Uploaded_File\NIPS-2017-attention-is-all-you-need-Paper.pdf'
loader = PyPDFLoader(file_path)
documents = loader.load()
print("#PDF Loader")

#RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print("#RecursiveCharacterTextSplitter")

#EmbeddingFunction
YOUR_API_KEY = os.getenv("GOOGLE_AI_STUDIO")
os.environ["GOOGLE_API_KEY"] = YOUR_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
print("#EmbeddingFunction")

#Add to Chroma DB Locally
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="Ascension_DB",
    embedding_function=embeddings,
    persist_directory="./DB",
)
vector_store.add_documents(documents=texts)
print("#Add to Chroma DB Locally")