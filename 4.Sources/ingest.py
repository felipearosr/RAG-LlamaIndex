import os
import openai
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

documents = SimpleDirectoryReader("./data").load_data()
index = GPTVectorStoreIndex.from_documents(documents)
index.storage_context.persist()