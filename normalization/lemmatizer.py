import spacy
from langchain_core.documents import Document

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def lemmatize_documents(documents):
    """
    Apply spaCy lemmatization to LangChain documents.
    """

    normalized_documents = []

    for doc in documents:
        text = doc.page_content
        spacy_doc = nlp(text)

        lemmas = []
        for token in spacy_doc:
            # Skip spaces
            if token.is_space:
                continue
            lemmas.append(token.lemma_)

        normalized_text = " ".join(lemmas)

        normalized_documents.append(
            Document(
                page_content=normalized_text,
                metadata=doc.metadata
            )
        )

    return normalized_documents
