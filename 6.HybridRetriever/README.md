# HybridRetriever

Added HybridRetriever, in order for this to work you will need to recreate your Pinecone DB by uncommenting `crete_index()`, you can do this manually too in the Pinecone website, just make sure you have the same dimesions and metric. 

You can read more information about the model, embeddings and dimension in the file `5.Pinecone/README.md`

![Alt Text](images/RAGSources.png)
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
