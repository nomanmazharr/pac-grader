from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
together_api = os.getenv("TOGETHER_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")


llm_grader = ChatTogether(
    # model="openai/gpt-oss-20b",
    model = "talhapacahmed_9091/gpt-oss-20b-78eb5223-03bd6cc1",
    temperature=0,
    api_key= together_api,
    max_tokens=80000
)

# llm_grader = ChatTogether(
#     model="openai/gpt-oss-20b",
#     temperature=0,
#     api_key= together_api,
#     max_tokens=80000
# )

llm = ChatTogether(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key= together_api,
    max_tokens=80000
)