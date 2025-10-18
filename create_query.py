import chromadb
from dotenv import load_dotenv
import os
load_dotenv()
from pprint import pprint

#Retrieve PersistentClient local DB
client = chromadb.PersistentClient(path="./DB")
collection = client.get_or_create_collection(name="Ascension_DB")
print("#Retrieve PersistentClient local DB")

#EmbeddingFunction
YOUR_API_KEY = os.getenv("GOOGLE_AI_STUDIO")
os.environ["GOOGLE_API_KEY"] = YOUR_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
print("#EmbeddingFunction")

#Retrieve from Chroma DB Locally
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="Ascension_DB",
    embedding_function=embeddings,
    persist_directory="./DB")
user_input = input('Please ask your question: ')
results = vector_store.similarity_search_by_vector(embedding=embeddings.embed_query(user_input), k=3)
pprint("#Retrieve from Chroma DB Locally")

#Invoke Chain
from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
output = llm.invoke(f"Summarize this content and answer as per user question just give the ans nothing else. Question:{user_input} Content:{results}")
print(output)