# Basic Doc ChatBot

Welcome to GPT Documents, a basic OpenAI document chatbot powered by llama index and chainlit. This is the first and most basic form of a chatbot with documents.

![](https://github.com/felipearosr/GPT-Documents/blob/main/1.Streaming%20-%20Memory%20-%20Sources/images/RAG.gif)


## Table of Contents

1. [Installation](#installation")
2. [Usage](#usage)
3. [Streaming](#streaming)
3. [Memory](#memory)
3. [Sources](#sources)

## Installation <a name="installation"></a>

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

## Usage <a name="usage"></a>

Once the setup is complete, launch the chainlit app using the following command:

```shell
chainlit run main.py
```

## Streaming <a name="streaming"></a>

### What is streaming?


### How do we implement it?

```python
Settings.llm = OpenAI(
        model="gpt-3.5-turbo", temperature=0.1, max_tokens=1024, streaming=True
)
```





## Memory <a name="memory"></a>

### What is memory?

### How do we implement it?
```python
@cl.on_chat_start
async def start():
    # ...
    message_history = [] # create an empty list to store the message history
    cl.user_session.set("message_history", message_history) # set it to user session
    # ...
```

```python
@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history") # get it from user session
    prompt_template = "Previous messages:\n"
    # ...
    user_message = message.content

    for message in message_history:
        prompt_template += f"{message['author']}: {message['content']}\n"
    prompt_template += f"Human: {user_message}"
    # ...
    message_history.append({"author": "Human", "content": user_message})
    message_history.append({"author": "AI", "content": response_message.content})
    message_history = message_history[-4:]
    cl.user_session.set("message_history", message_history)
```

## Sources <a name="sources"></a>

### What are sources?

Sources refer to the documents or materials returned by the retrieval system, which provide the foundation for the answers to your queries. They offer a transparent way to verify the origin of the information used by the language model to generate its responses.

### How do we implement it?

```python
@cl.on_message
async def main(message: cl.Message):
    # rest of your code
    label_list = []
    count = 1

    for sr in response.source_nodes:
        elements = [
            cl.Text(
                name="S" + str(count),
                content=f"{sr.node.text}",
                display="side",
                size="small",
            )
        ]
        response_message.elements = elements
        label_list.append("S" + str(count))
        await response_message.update()
        count += 1
    response_message.content += "\n\nSources: " + ", ".join(label_list)
    await response_message.update()
```
