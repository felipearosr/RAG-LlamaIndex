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


class Config:
    def __init__(self):
        self.openai_api_key = self._load_env_var("OPENAI_API_KEY")
        self.cohere_api_key = self._load_env_var("COHERE_API_KEY")
        self.pinecone_api_key = self._load_env_var("PINECONE_API_KEY")
        self.model = os.getenv("MODEL", "gpt-4-0125-preview")
        self.embedding = os.getenv("EMBEDDING", "text-embedding-3-large")
        self.temperature = 0.1
        self.num_output = 1024
        self.context_window = 128000

    @staticmethod
    def _load_env_var(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise EnvironmentError(f"Missing required environment variable: {name}")
        return value


def set_settings(config: Config):
    """
    Sets the settings for the application using environment variables.
    Includes settings for LLM, embedding model, output number, and context window size.
    """
    model = os.getenv("MODEL", "gpt-4-0125-preview")
    embed_model = os.getenv("EMBEDDING", "text-embedding-3-large")

    Settings.llm = OpenAI(
        temperature=config.temperature,
        model=config.model,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=config.embedding,
    )
    Settings.num_output = config.num_output
    Settings.context_window = config.context_window


def get_index(config: Config):
    """
    Initializes and returns the Pinecone index used for vector storage.
    """
    pc = Pinecone(api_key=config.pinecone_api_key)
    pinecone_index = pc.Index("pinecone-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def get_query_engine(index, config: Config):
    """
    Creates and returns a query engine with specified settings, including the step
    decomposition transform and the reranker.
    Args:
        index: The VectorStoreIndex object to use for querying.
    """
    step_decompose_transform = StepDecomposeQueryTransform(verbose=True)

    reranker = CohereRerank(api_key=config.cohere_api_key, top_n=3)
    query_engine = index.as_query_engine(
        similarity_top_k=6,
        vector_store_query_mode="hybrid",
        node_postprocessors=[reranker],
        query_transform=step_decompose_transform,
        response_synthesizer_mode="refine",
    )
    return query_engine


def generate_questions():
    """
    Generates a list of questions from documents stored in a specified directory.
    Returns:
        A list of question strings generated from the document data.
    """
    documents = SimpleDirectoryReader("./data").load_data()
    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        num_questions_per_chunk=2,  # set the number of questions per nodes
    )
    rag_dataset = dataset_generator.generate_questions_from_nodes()
    questions = [e.query for e in rag_dataset.examples]
    return questions


async def evaluate(questions, query_engine):
    """
    Evaluates the set of questions using the provided query engine.
    Args:
        questions: A list of question strings to evaluate.
        query_engine: The query engine to use for evaluation.
    Returns:
        The results of the evaluation.
    """
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


async def main():
    """
    Oorchestrates the loading of environment variables, setting configurations,
    initializing index and query engine, generating or loading questions,
    and evaluating them.
    """
    config = Config()
    set_settings(config)

    index = get_index(config)
    query_engine = get_query_engine(index, config)

    questions_file = "questions.json"

    try:
        with open(questions_file, "r") as f:
            questions = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        questions = generate_questions()
        with open(questions_file, "w") as f:
            json.dump(questions, f)

    eval_results = await evaluate(questions, query_engine)

    print(eval_results)
