# ChatBot with streaming, memory and sources

Embarking on the creation of an advanced Retrieval-Augmented Generation (RAG) system marks a significant first step towards innovative chatbot development. This foundational version incorporates three critical features:

- **Streaming:** Enhance user experience with fast, real-time answers as the chatbot generates responses on-the-fly, reducing wait times.
- **Memory:** Facilitate natural, conversational interactions by enabling the chatbot to recall previous parts of the conversation, adding context and relevance to the dialogue.
- **Sources:** Increase transparency and trust by clearly indicating the origin of the chatbot's answers, allowing users to understand where the information is coming from.

These functionalities are powered by technologies like the Llama-index and Chainlit, setting the stage for a more intuitive, responsive, and informed chatbot experience.

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

### Understanding Streaming in LLMs

Streaming is a feature that enables real-time delivery of responses from the language learning model (LLM) as they are being generated. This process significantly reduces response latency by allowing immediate display of each part of the answer, token by token, as it is streamed from the LLM. This means users do not have to wait for the entire response to be composed and sent before beginning to read the answer, facilitating a smoother and faster interaction.


### How do we implement it?

```python
@cl.on_chat_start
async def start():
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo", temperature=0.1, max_tokens=1024, streaming=True
    )
```





## Memory <a name="memory"></a>

### Exploring Memory in LLMs

Memory in llms is a feature we integrate to enhance their ability to maintain and recall the history of interactions with users. This functionality enriches the conversational experience by allowing the model to reference previous exchanges and build on them, creating a more coherent and contextually relevant dialogue.

### How do we implement it?
```python
@cl.on_chat_start
async def start():
    # ...
    # create an empty list to store the message history
    message_history = []
    # set message_history to user_session
    cl.user_session.set("message_history", message_history) 
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

### What are Sources?

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
