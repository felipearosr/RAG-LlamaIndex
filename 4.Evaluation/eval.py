import os
import json
import openai

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    BatchEvalRunner,
    RelevancyEvaluator,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)


openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")


def set_settings():
    model = os.getenv("MODEL", "gpt-4-0125-preview")
    embed_model = os.getenv("EMBEDDING", "text-embedding-3-large")

    Settings.llm = OpenAI(
        temperature=0.1,
        model=model,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=embed_model,
    )
    Settings.num_output = 1024
    Settings.context_window = 128000


def get_index():
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")

        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index("pinecone-index")
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
    except (KeyError, ValueError):
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    return index


def get_query_engine(index):
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    step_decompose_transform = StepDecomposeQueryTransform(verbose=True)

    reranker = CohereRerank(api_key=cohere_api_key, top_n=3)
    query_engine = index.as_query_engine(
        similarity_top_k=6,
        vector_store_query_mode="hybrid",
        node_postprocessors=[reranker],
        query_transform=step_decompose_transform,
        response_synthesizer_mode="refine",
    )
    return query_engine


def generate_questions():
    documents = SimpleDirectoryReader("./data").load_data()
    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        num_questions_per_chunk=2,  # set the number of questions per nodes
    )
    rag_dataset = dataset_generator.generate_questions_from_nodes()
    questions = [e.query for e in rag_dataset.examples]
    return questions


async def evaluate(questions, query_engine):
    faithfulness_evaluator = FaithfulnessEvaluator()
    relevancy_evaluator = RelevancyEvaluator()
    runner = BatchEvalRunner(
        {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
        workers=8,
    )
    eval_results = await runner.aevaluate_queries(
        query_engine=query_engine, queries=questions
    )
    return eval_results


def main():
    set_settings()

    index = get_index()
    query_engine = get_query_engine(index)

    questions_file = "questions.json"

    try:
        with open(questions_file, "r") as f:
            questions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        questions = generate_questions()
        with open(questions_file, "w") as f:
            json.dump(questions, f)

    eval_results = evaluate(questions, query_engine)

    print(eval_results)
