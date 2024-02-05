# Advanced Ingestion with llmsherpa

Improved ingestion of PDF files with [llmsherpa](https://github.com/nlmatics/llmsherpa). This includes features like:


1. Sections and subsections along with their levels.
2. Paragraphs - combines lines.
3. Links between sections and paragraphs.
4. Tables along with the section the tables are found in.
5. Lists and nested lists.
6. Join content spread across pages.
6. Removal of repeating headers and footers.
7. Watermark removal.

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
