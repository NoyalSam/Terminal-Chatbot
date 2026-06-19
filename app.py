from config_validator import validate_config

from loaders.document_loader import load_document
from normalization import lemmatize_documents 
from chunkers import cosine_chunk_documents
 
from embeddings.embedding_factory import get_embeddings
from embeddings.llm_factory import get_llm

from vectorstore.faiss_store import create_vector_store, get_retriever

from chains.rag_chain import build_rag_chain
from pydantic_model import build_pydantic_chain

from tools.search_tool import create_search_tool
from tools.calculator_tool import calculator

from agents.rag_agent import build_agent

from langchain_core.runnables.history import RunnableWithMessageHistory
from memory.memory_manager import get_session_history


def main():

    validate_config()

    print("Loading documents...")

    documents = load_document("data/Noyal_Sam_D.pdf")

    print("Normalizing documents...")

    normalized_documents = lemmatize_documents(documents)

    print("Chunking documents...")

    chunks = cosine_chunk_documents(normalized_documents)

    print("Creating vector store...")

    embeddings = get_embeddings()
    vectorstore = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    llm = get_llm()

    pydantic_chain = build_pydantic_chain(llm)
    rag_chain = build_rag_chain(retriever, llm)

    search_tool = create_search_tool(rag_chain)

    tools = [search_tool, calculator]

    agent = build_agent(llm, tools)

    # Memory wrapper
    agent_with_memory = RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key="messages",
    )

    print("\nRAG Agent Ready\n")

    session_id = "user_1"
    chat_history = ""

    while True:

        query = input("User: ")

        if query.lower() == "exit":
            break

        # Rewrite the query into a standalone question (Pydantic structured output)
        parent_result = pydantic_chain.invoke({
            "question": query,
            "chat_history": chat_history
        })

        standalone_query = parent_result.standalone_question

        response = agent_with_memory.invoke(
            {
                "messages": [
                    ("user", standalone_query)
                ]
            },
            config={"configurable": {"session_id": session_id}},
        )

        answer = extract_text(response)

        print("Assistant:", answer)

        chat_history += f"User: {query}\nAssistant: {answer}\n"


def extract_text(response):

    content = response["messages"][-1].content

    if isinstance(content, list):
        return content[0].get("text", "")

    return content


if __name__ == "__main__":
    main()
