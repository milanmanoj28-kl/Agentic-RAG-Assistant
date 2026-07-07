import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from agent import agent, vector_store
from load_documents import split_documents
from langchain_community.document_loaders import PyPDFLoader, TextLoader

app = FastAPI(title="RAG Agent API")

DOCUMENTS_FOLDER = "documents"
conversation_history = []


class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


@app.get("/")
def root():
    return {"message": "RAG Agent API is running. Use POST /chat to ask questions, POST /upload to add documents."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    conversation_history.append({"role": "user", "content": request.question})
    result = agent.invoke({"messages": conversation_history})
    ai_message = result["messages"][-1]
    conversation_history.append({"role": "assistant", "content": ai_message.content})
    return ChatResponse(answer=ai_message.content)


@app.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    # Save the uploaded file into the documents folder
    filepath = os.path.join(DOCUMENTS_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load just this one file
    if file.filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif file.filename.endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        return UploadResponse(filename=file.filename, chunks_added=0, message="Unsupported file type. Use PDF or TXT.")

    docs = loader.load()
    chunks = split_documents(docs)

    # Add these new chunks into the existing vector store
    vector_store.add_documents(chunks)

    return UploadResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        message=f"'{file.filename}' successfully processed and added to the knowledge base."
    )