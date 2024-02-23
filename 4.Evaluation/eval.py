import argparse
import os
import json
import openai
import asyncio
import csv

from dotenv import load_dotenv
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    BatchEvalRunner,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)


class Config:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = self._load_env_var("OPENAI_API_KEY")
        self.cohere_api_key = self._load_env_var("COHERE_API_KEY")
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


def get_index(documents):
    """
    Initializes and returns the Pinecone index used for vector storage.
    """
    return VectorStoreIndex.from_documents(documents)


def get_documents(data_directory):
    """
    Loads and returns the documents from a specified directory.
    """
    if not os.path.exists(data_directory):
        raise FileNotFoundError(
            f"The data directory '{data_directory}' does not exist."
        )
    return SimpleDirectoryReader(data_directory).load_data(show_progress=True)


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
        # vector_store_query_mode="hybrid",
        node_postprocessors=[reranker],
        query_transform=step_decompose_transform,
        response_synthesizer_mode="refine",
    )
    return query_engine


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


async def generate_questions(documents):
    """
    Generates a list of questions from documents stored in a specified directory.
    Returns:
        A list of question strings generated from the document data.
    """
    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        num_questions_per_chunk=2,  # set the number of questions per nodes
    )
    rag_dataset = await dataset_generator.agenerate_questions_from_nodes()
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
    relevancy_evaluator = RelevancyEvaluator()
    correctness_evaluator = CorrectnessEvaluator()
    faithfulness_evaluator = FaithfulnessEvaluator()

    runner = BatchEvalRunner(
        {
            "relevancy": relevancy_evaluator,
            "correctness": correctness_evaluator,
            "faithfulness": faithfulness_evaluator,
        },
        workers=8,
    )
    eval_results = await runner.aevaluate_queries(
        query_engine=query_engine, queries=questions
    )
    return eval_results


async def main():
    """
    Orchestrates the loading of environment variables, setting configurations,
    initializing index and query engine, generating or loading questions,
    and evaluating them. Supports generating questions based on a command line flag.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate new questions before evaluation",
    )
    args = parser.parse_args()

    try:
        config = Config()
        print("Environment variables loaded successfully.")

        set_settings(config)
        print("Settings set successfully.")

        print("Loading documents...")
        documents = get_documents("./data")

        index = get_index(documents)
        query_engine = get_query_engine(index, config)
        print("Query engine initialized successfully.")

        questions_file = "questions.json"

        if args.generate:
            print("Flag for generating new questions detected.")
            rag_dataset = await generate_questions(documents)
            print("Generated new questions.")
            with open(questions_file, "w") as f:
                json.dump(rag_dataset, f)
            print(f"Questions saved to {questions_file}.")
        else:
            print("Checking for existing questions...")
            try:
                with open(questions_file, "r") as f:
                    content = f.read().strip()  # Read and strip whitespace
                    if content:  # Check if the content is non-empty
                        rag_dataset = json.loads(content)
                        print("Loaded questions from file.")
                    else:
                        raise json.JSONDecodeError("File is empty", content, 0)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading questions: {e}")
                rag_dataset = await generate_questions(documents)
                print("Generated new questions due to error loading existing ones.")
                with open(questions_file, "w") as f:
                    json.dump(rag_dataset, f)
                print(f"Questions saved to {questions_file}.")

        print("Evaluating questions...")
        eval_results = await evaluate(rag_dataset, query_engine)
        print("Evaluation completed.")

        #print("Evaluation Results:", eval_results)

        score = get_eval_results("correctness", eval_results)
        print(f"Correctness Score: {score}")
        score = get_eval_results("relevancy", eval_results)
        print(f"Relevancy Score: {score}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
