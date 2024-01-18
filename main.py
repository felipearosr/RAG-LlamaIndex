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

    await cl.Message(author="Assistant", content="Hello! Im an AI assistant. How may I help you?").send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    msg = cl.Message(content="", author="Assistant")

    res = query_engine.query(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
