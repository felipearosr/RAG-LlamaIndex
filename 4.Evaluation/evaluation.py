import os
import csv
import json
import openai
import asyncio
import argparse
import pandas as pd

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

from main import get_query_engine, get_index
from generation import handle_question_generation


class Config:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = self._load_env_var("OPENAI_API_KEY")
        self.cohere_api_key = self._load_env_var("COHERE_API_KEY")
        self.model = os.getenv("MODEL", "gpt-3.5-turbo-0125") # gpt-4-0125-preview
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


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


def write_eval_results_to_csv_with_pandas(eval_results):
    """
    Writes merged evaluation results to a CSV file using Pandas. This function merges results from 'correctness',
    'relevancy', and 'faithfulness' categories, specifically their scores, into one record.
    Common attributes like 'query', 'response', etc., are maintained. Scores for each category are separated into distinct columns.
    The CSV file is saved into an 'output' folder. If the folder does not exist, it is created.
    If a file with the name 'eval_results.csv' exists, a new file is created with a timestamp to avoid overwriting.
    """
    reshaped_results = []

    # Define the new columns, including separate score columns for each category and other common attributes
    columns = [
        "Query",
        "Response",
        "Contexts",
        "Passing",
        "Feedback",
        "PairwiseSource",
        "InvalidResult",
        "InvalidReason",
        "Correctness_Score",
        "Relevancy_Score",
        "Faithfulness_Score",
    ]

    # Iterate through each set of evaluations assuming they are of the same length and aligned
    for index in range(
        len(eval_results["correctness"])
    ):  # Assumes all lists have the same length
        new_record = {col: None for col in columns}

        # Merge records from each category for the same index
        for category in ["correctness", "relevancy", "faithfulness"]:
            eval_result = eval_results[category][index]

            # Convert EvaluationResult to dictionary if necessary
            result_data = (
                eval_result.dict() if hasattr(eval_result, "dict") else eval_result
            )

            # Populate the new record, ensuring scores are assigned to their specific columns
            for key, value in result_data.items():
                if key == "score":
                    new_record[f"{category.capitalize()}_Score"] = (
                        value  # Category-specific score
                    )
                else:
                    # Ensure capitalization matches the column names and prevent overwriting of existing values
                    proper_key = key[0].upper() + key[1:] if key else key
                    if proper_key in new_record and new_record[proper_key] is None:
                        new_record[proper_key] = value

        # Add the merged record to the reshaped results
        reshaped_results.append(new_record)

    # Write the reshaped results to a CSV file
    if reshaped_results:
        import pandas as pd
        import os
        from datetime import datetime

        df = pd.DataFrame(reshaped_results, columns=columns)
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        filename = "eval_results.csv"
        full_path = os.path.join(output_dir, filename)
        if os.path.exists(full_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = os.path.join(output_dir, f"eval_results_{timestamp}.csv")
        df.to_csv(full_path, index=False)
        print(f"Results written to {full_path}")
    else:
        print("No data to write to CSV.")


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
    parser = argparse.ArgumentParser(description="Run components based on flags.")
    # parser.add_argument('--eval', action='store_true', help='Run the evaluation component')
    parser.add_argument(
        "--gen", action="store_true", help="Run the generation component"
    )
    parser.add_argument("--tune", action="store_true", help="Run the tuning component")
    args = parser.parse_args()

    try:
        config = Config()
        print("Environment variables loaded successfully.")

        index = get_index()
        query_engine = get_query_engine(index, config)
        print("Query engine initialized successfully.")

        questions_file = "questions.json"
        """
        if args.eval:
            print("Running evaluation")
        """
        missing_questions = False

        if args.tune:
            print("Running tuning")
            return

        """
        quesiton generation check,
        """
        try:
            with open(query_engine, "r") as f:
                content = f.read().strip()
                if content:
                    rag_dataset = json.loads(content)
                    print("Loaded questions from file.")
                else:
                    raise json.JSONDecodeError("File is empty", content, 0)
        except:
            missing_questions = True

        # nice
        if args.gen or missing_questions:
            if not args.gen and missing_questions:
                print("Running generation")
                rag_dataset = await handle_question_generation(questions_file)
            else:
                print("Overriting questions")
                rag_dataset = await handle_question_generation(questions_file)
                return

        print("Evaluating questions...")
        eval_results = await evaluate(rag_dataset, query_engine)
        print("Evaluation completed.")

        write_eval_results_to_csv_with_pandas(eval_results)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
