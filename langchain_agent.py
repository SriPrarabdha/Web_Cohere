from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
import os
import pinecone
from pinecone_text.sparse import SpladeEncoder

index_name = "criminal-laws"
pinecone.init(
    api_key="f112db94-1b02-44ec-b1d7-a4cf165fad28",
    environment="us-east1-gcp"  
)

embeddings = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-tas-b")
  
# use default tf-idf values
splade_encoder = SpladeEncoder()

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade_encoder, index=index)
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

from langchain.chains import RetrievalQA

dnd_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

os.environ["SERPAPI_API_KEY"] = "b5e4eee837ecc4cb0336916bb63ccc8e6158510787b74dae09c01504eb045b4c"
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or when you get no results from criminal-laws tool",
    ),
    Tool(
        name="web",
        func=dnd_qa.run,
        description="useful for when you need to know something about website scraped. Input: an objective to know more about a website and it's applications. Output: Correct interpretation of the question. Please be very clear what the objective is!",
    ),
]


prefix = """You are an AI who is a assistant to a lawyer your only objective is to assist the user in solving any query related to law: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"]
)