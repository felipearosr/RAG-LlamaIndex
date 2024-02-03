import os
import openai

from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from multiprocessing import freeze_support

from llama_index import (
    OpenAIEmbedding,
    SimpleDirectoryReader
)
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import PineconeVectorStore

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

MODEL = "gpt-4-0125-preview"  # "gpt-3.5-turbo" - "gpt-3.5-turbo-0125" - OLD "gpt-3.5-turbo"
EMBEDDING = "text-embedding-3-large"  # "text-embedding-3-small" - OLD "text-embedding-ada-002"

pc = Pinecone(api_key=pinecone_api_key)

num_workers = min(4, os.cpu_count())

"""
pc.create_index(
    name="rag-index",
    dimension=3072,
    metric="cosine",
    spec=PodSpec(environment="gcp-starter"),
)
"""
pinecone_index = pc.Index("rag-index")
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
)

def run_pipeline():
    llm = OpenAI(temperature=0.1, model=MODEL, max_tokens=1024)

    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=126),
            #TitleExtractor(llm=llm, num_workers=num_workers),
            #QuestionsAnsweredExtractor(questions=3, llm=llm, num_workers=num_workers),
            #SummaryExtractor(summaries=["prev", "self"], llm=llm, num_workers=num_workers),
            #KeywordExtractor(keywords=5, llm=llm, num_workers=num_workers),
            OpenAIEmbedding(model=EMBEDDING),
        ],
        vector_store=vector_store,
    )

    pipeline.run(documents=documents, show_progress=True, num_workers=num_workers)


if __name__ == "__main__":
    freeze_support()
    run_pipeline()
