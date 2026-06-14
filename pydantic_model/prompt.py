PARENT_QUESTION_PROMPT = """
Given the conversation history and a follow-up question, rewrite the
follow-up question into a standalone "parent" question that contains
all the necessary context to be understood without the chat history.

If the follow-up question is already standalone, return it unchanged.

Chat History:
{chat_history}

Follow-up Question:
{question}

Standalone Question:
"""
