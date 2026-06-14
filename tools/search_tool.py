from langchain.tools import tool


def create_search_tool(rag_chain):

    @tool
    def search_documents(query: str) -> str:
        """Search information from the document"""
        return rag_chain.invoke(query)

    return search_documents