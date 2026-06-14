from langchain.agents import create_agent


def build_agent(llm, tools):
    """
    Build a tool-calling agent using LangChain's modern API.
    """

    agent = create_agent(
        model=llm,
        tools=tools
    )

    return agent