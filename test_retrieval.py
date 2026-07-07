from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "chroma_db"

def test_search(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    results = vector_store.similarity_search(query, k=2)  # top 2 matching chunks

    print(f"\nQuery: {query}")
    print("-" * 50)
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content)
        print(f"(Source: {doc.metadata.get('source', 'unknown')})")


if __name__ == "__main__":
    test_search("What projects has Milan worked on?")