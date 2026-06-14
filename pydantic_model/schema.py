from pydantic import BaseModel, Field


class ParentQuestion(BaseModel):
    """
    Structured output for the parenting model.
    Rewrites a follow-up question into a standalone "parent" question.
    """

    standalone_question: str = Field(
        description="The follow-up question rewritten as a standalone "
                     "question that includes all necessary context from "
                     "the chat history."
    )

    is_followup: bool = Field(
        description="True if the original question depended on chat "
                     "history context (e.g. used pronouns like 'it', "
                     "'that', 'he'), False if it was already standalone."
    )
