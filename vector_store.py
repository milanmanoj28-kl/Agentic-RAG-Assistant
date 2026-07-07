from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from load_documents import load_documents, split_documents

PERSIST_DIRECTORY = "chroma_db"

def create_vector_store():
    """Load documents, split them, embed them, and store in ChromaDB."""

    #split documents
    docs = load_documents()
    chunks = split_documents(docs)

    #the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: Create the vector store and save it to disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"Vector store created and saved to '{PERSIST_DIRECTORY}'")
    return vector_store


if __name__ == "__main__":
    create_vector_store()