from langchain_experimental.text_splitter import SemanticChunker


def semantic_chunk_documents(documents, embeddings):
    """
    Perform semantic chunking based on embedding similarity.
    Splits document at topic boundaries instead of fixed characters.
    """

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="gradient",   # Detects topic shifts
        buffer_size=1,                          # Adds neighboring sentences
        min_chunk_size=200                      # Avoids very small chunks
    )

    chunks = splitter.split_documents(documents)

    return chunks