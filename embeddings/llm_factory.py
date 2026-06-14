from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings


def get_llm():

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0.2
    )

    return llm