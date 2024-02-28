import os
import csv
import json
import openai

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_dataset.generator import RagDatasetGenerator


def get_documents(data_directory):
    """
    Loads and returns the documents from a specified directory.
    """
    if not os.path.exists(data_directory):
        raise FileNotFoundError(
            f"The data directory '{data_directory}' does not exist."
        )
    return SimpleDirectoryReader(data_directory).load_data(show_progress=True)

async def generate_questions():
    """
    Generates a list of questions from documents stored in a specified directory.
    Returns:
        A list of question strings generated from the document data.
    """
    data_directory = "./data"
    documents = get_documents(data_directory)

    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        num_questions_per_chunk=3,  # set the number of questions per nodes
    )
    rag_dataset = await dataset_generator.agenerate_questions_from_nodes()
    questions = [e.query for e in rag_dataset.examples]
    return questions

async def handle_question_generation(questions_file):
    """
    Handles the generation of new questions or the loading of existing questions based on command line arguments.
    Args:
        args: Command line arguments.
        documents: The document data used for question generation.
        questions_file: The file where questions are saved or loaded from.
    Returns:
        A dataset containing the questions.
    """
    #if args.generate:
    #print("Flag for generating new questions detected.")
    rag_dataset = await generate_questions()
    print("Generated new questions.")
    with open(questions_file, "w") as f:
        json.dump(rag_dataset, f)
    print(f"Questions saved to {questions_file}.")
    """
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
    """
    return rag_dataset