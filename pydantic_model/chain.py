from langchain_core.prompts import PromptTemplate

from pydantic_model.prompt import PARENT_QUESTION_PROMPT
from pydantic_model.schema import ParentQuestion


def build_pydantic_chain(llm):
    """
    Build an LCEL chain that rewrites a follow-up question into a
    standalone "parent" question using chat history context.

    Uses a Pydantic model (ParentQuestion) for structured output.

    Input: {"question": str, "chat_history": str}
    Output: ParentQuestion(standalone_question: str, is_followup: bool)
    """

    prompt = PromptTemplate.from_template(PARENT_QUESTION_PROMPT)

    structured_llm = llm.with_structured_output(ParentQuestion)

    pydantic_chain = (
        prompt
        | structured_llm
    )

    return pydantic_chain
