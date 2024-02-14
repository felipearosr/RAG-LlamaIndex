import os
import openai
import tiktoken
import nest_asyncio
import numpy as np
import pandas as pd

from llama_index import (
    ServiceContext,
    OpenAIEmbedding,
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.vector_stores import PineconeVectorStore
from llama_index.callbacks import CallbackManager, TokenCountingHandler

from trulens_eval import Tru, TruLlama, Feedback, feedback
from trulens_eval import OpenAI as fOpenAI
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness

from pinecone import Pinecone
from dotenv import load_dotenv

nest_asyncio.apply()

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

MODEL = "gpt-4-0125-preview"
EMBEDDING = "text-embedding-3-large"

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model(MODEL).encode,
    verbose=True,
)

pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index("pinecone-index")
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
)
service_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0.1, model=MODEL, max_tokens=3072, streaming=True),
    embed_model=OpenAIEmbedding(model=EMBEDDING),
    callback_manager=CallbackManager([token_counter]),
)
set_global_service_context(service_context)

index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(
    similarity_top_k=4,
    vector_store_query_mode="hybrid",
)

provider = fOpenAI()
grounded = Groundedness(groundedness_provider=provider)

context = TruLlama.select_source_nodes().node.text

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()
f_qs_relevance = (
    Feedback(provider.qs_relevance, name="Context Relevance")
    .on_input()
    .on(context)
    .aggregate(np.mean)
)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(context)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)
tru = Tru()
tru.reset_database()

tru_recorder = TruLlama(
    app=query_engine,
    app_id="Asistente Sura v1.0 + Prompt",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ],
)

eval_questions = []
with open("eval_questions.txt", "r") as file:
    for line in file:
        item = line.strip()
        eval_questions.append(item)


for question in eval_questions:
    with tru_recorder as recording:
        query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()


import pandas as pd

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]

tru.get_leaderboard(app_ids=[])

tru.run_dashboard()
