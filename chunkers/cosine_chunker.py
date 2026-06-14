import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_model = SentenceTransformer("all-MiniLM-L6-v2")


def dynamic_semantic_chunk(
    text,
    window_size=3,
    min_chunk_sentences=3,
    max_chunk_sentences=15
):
    """
    Split text into chunks based on cosine similarity drops between
    rolling-window sentence embeddings.
    """

    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= min_chunk_sentences:
        return [text]

    embeddings = _model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = []
    for i in range(len(sentences) - 1):
        left_start = max(0, i - window_size + 1)
        left_end = i + 1
        right_start = i + 1
        right_end = min(len(sentences), i + window_size + 1)

        left_vec = embeddings[left_start:left_end].mean(axis=0)
        right_vec = embeddings[right_start:right_end].mean(axis=0)

        similarity = np.dot(left_vec, right_vec)
        similarities.append(similarity)

    similarities = np.array(similarities)
    threshold = similarities.mean() - similarities.std()

    chunks = []
    current_chunk = [sentences[0]]

    for i, similarity in enumerate(similarities):
        should_split = (
            similarity < threshold
            and len(current_chunk) >= min_chunk_sentences
        )
        force_split = len(current_chunk) >= max_chunk_sentences

        if should_split or force_split:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def cosine_chunk_documents(
    documents,
    window_size=3,
    min_chunk_sentences=3,
    max_chunk_sentences=15
):
    """
    Apply cosine-similarity-based dynamic chunking to a list of
    LangChain Documents. Replaces SemanticChunker (gradient-based)
    with sentence-window cosine similarity splitting.
    """

    chunked_documents = []

    for doc in documents:
        text_chunks = dynamic_semantic_chunk(
            doc.page_content,
            window_size=window_size,
            min_chunk_sentences=min_chunk_sentences,
            max_chunk_sentences=max_chunk_sentences
        )

        for chunk_text in text_chunks:
            chunked_documents.append(
                Document(
                    page_content=chunk_text,
                    metadata=doc.metadata
                )
            )

    return chunked_documents
