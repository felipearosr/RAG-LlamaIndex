import os
import openai
import chainlit as cl

from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)

openai.api_key = os.environ.get("OPENAI_API_KEY")


@cl.cache
def load_context():
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage",
    )
    index = load_index_from_storage(storage_context)
    return index


@cl.on_chat_start
async def start():
    index = load_context()

    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            temperature=0.1, model="gpt-3.5-turbo", max_tokens=1024, streaming=True
        ),
        text_splitter=SentenceSplitter(chunk_size=512, chunk_overlap=126),
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )
    query_engine = index.as_query_engine(
        service_context=service_context, streaming=True, similarity_top_k=2
    )
    cl.user_session.set("query_engine", query_engine)

    message_history = []
    cl.user_session.set("message_history", message_history)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    message_history = cl.user_session.get("message_history")
    prompt_template = "Previous messages:\n"

    response_message = cl.Message(content="", author="Assistant")

    user_message = message.content

    for message in message_history:
        prompt_template += f"{message['author']}: {message['content']}\n"
    prompt_template += f"Human: {user_message}"

    response = await cl.make_async(query_engine.query)(prompt_template)

    for token in response.response_gen:
        await response_message.stream_token(token)
    if response.response_txt:
        response_message.content = response.response_txt
    await response_message.send()

    message_history.append({"author": "Human", "content": user_message})
    message_history.append({"author": "AI", "content": response_message.content})
    message_history = message_history[
        -6:
    ]  # This keeps only the last 3 pairs of messages
    cl.user_session.set("MESSAGE_HISTORY", message_history)

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
