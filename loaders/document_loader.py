from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(path: str): #path: str it recevie string only

    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)

    elif path.endswith(".txt"):
        loader = TextLoader(path)

    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()

    return documents