# Evaluation, generation and optimization

> [!IMPORTANT] 
> Evaluation and optimization is untested and likely unfinished.

## Table of Contents

1. [Installation Instructions](#installation-instructions)
2. [Usage](#usage)
3. [Generation](#generation)

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

## Generation

### Why is it important?

### What are we generating?

### How do we implement it?

```python


```

## Tested\
<table>
<tr>
<td align="left">Left text</td>
<td align="right">Right text</td>
</tr>
</table>
|a|   |b|
|-----------|------------------------------------------------|------------|
- [x] Generation.    <span style="float: right;">March 14, 2024 </span>   
- [ ] Evaluation.    <span style="float: right;">Untested </span>
- [ ] Optimization.  <span style="float: right;">Untested </span>