import os
import openai

from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from multiprocessing import freeze_support

from llama_index.core import SimpleDirectoryReader, download_loader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_hub.smart_pdf_loader import SmartPDFLoader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

MODEL = "gpt-4-0125-preview"
EMBEDDING = "text-embedding-3-large"

pc = Pinecone(api_key=pinecone_api_key)

num_cores = os.cpu_count()
num_workers = min(4, num_cores)

"""
pc.create_index(
    name="rag-index",
    dimension=3072,
    metric="dotproduct",
    spec=PodSpec(environment="gcp-starter"),
)
"""

pinecone_index = pc.Index("rag-index")
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    add_sparse_vector=True,
)

UnstructuredReader = download_loader("UnstructuredReader")

def run_pipeline():
    llm = OpenAI(temperature=0.1, model=MODEL, max_tokens=1024)

    directory = "./data/"

    directory_reader = SimpleDirectoryReader(
        input_dir=directory,
        file_extractor={
            ".html": UnstructuredReader(),
            ".txt": UnstructuredReader()
        },
    )

    documents = directory_reader.load_data(show_progress=True)

    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)

    # List all files in the folder
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    # Load each PDF document and append it to the documents list
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        pdf_document = pdf_loader.load_data(pdf_path)
        documents += pdf_document


    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=126),
            TitleExtractor(llm=llm, num_workers=num_workers),
            QuestionsAnsweredExtractor(questions=3, llm=llm, num_workers=num_workers),
            SummaryExtractor(summaries=["prev", "self"], llm=llm, num_workers=num_workers),
            KeywordExtractor(keywords=5, llm=llm, num_workers=num_workers),
            OpenAIEmbedding(model=EMBEDDING)
        ],
        vector_store=vector_store,
    )

    pipeline.run(documents=documents, show_progress=True, num_workers=num_workers)

if __name__ == '__main__':
    freeze_support()
    run_pipeline()


