from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
together_api = os.getenv("TOGETHER_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=openai_api)

llm_grader = ChatOpenAI(
    model="gpt-5-2025-08-07",
    api_key=openai_api,
    temperature=0
)

llm = ChatOpenAI(
    model="gpt-5-mini-2025-08-07",
    api_key=openai_api,
    temperature=0
)
