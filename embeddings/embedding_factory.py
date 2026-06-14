from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings


def get_embeddings():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL
    )

    return embeddings