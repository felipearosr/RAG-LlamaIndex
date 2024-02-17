# Basic Doc ChatBot

Welcome to GPT Documents, a basic OpenAI document chatbot powered by llama index and chainlit. This is the first and most basic form of a chatbot with documents.

![](https://github.com/felipearosr/GPT-Documents/blob/main/1.Streaming%20-%20Memory%20-%20Sources/images/RAG.gif)

## Installation Instructions

Follow these steps to set up the GPT Documents chatbot on your local machine:

1. Create a conda environment:

   ```shell
   conda create -n rag python==3.11 -y && source activate rag
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Load your documents into the vector store by: 
    - Create a folder named 'data'.
    - Place your documents inside the 'data' folder.
    - Execute the 'ingest.py' script to initiate the loading process.

## Usage

Once the setup is complete, launch the chainlit app using the following command:

```shell
chainlit run -w main.py
```

Feel free to explore the functionalities and contribute to the development of this project. Your feedback and contributions are highly appreciated!
