import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_FOLDER = "documents"

def load_documents():
    """Load all PDF and text files from the documents folder."""
    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} document(s) from '{DOCUMENTS_FOLDER}'")
    return documents


def split_documents(documents):
    """Break documents into small overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      
        chunk_overlap=150,    
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)

    # Print the first chunk so you can see what it looks like
    if chunks:
        print("\n--- Example chunk ---")
        print(chunks[0].page_content)