import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tavily import TavilyClient

load_dotenv()

PERSIST_DIRECTORY = "chroma_db"

# ---- Set up the vector store (same as before) ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# ---- Tool 1: Document search ----
@tool
def document_search(query: str) -> str:
    """Search Milan's documents (CV, cover letters) for relevant information about his background, skills, projects, or experience."""
    results = vector_store.similarity_search(query, k=5)
    if not results:
        return "No relevant information found in the documents."
    return "\n\n".join([doc.page_content for doc in results])

# ---- Tool 2: Web search (using Tavily directly) ----
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for general questions not related to Milan's documents, like current events or general knowledge."""
    response = tavily_client.search(query, max_results=3)
    results = response.get("results", [])
    if not results:
        return "No web results found."
    return "\n\n".join([r.get("content", "") for r in results])

# ---- The LLM ----
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---- Build the agent (new LangChain 1.0 way) ----
agent = create_agent(
    model=llm,
    tools=[document_search, web_search],
    system_prompt="You are a helpful assistant. Use document_search for questions about Milan's background, CV, or experience. Use web_search for general questions. Answer in plain, natural sentences only. Do not use HTML tags, angle brackets, markdown formatting, or special symbols around any part of your answer — write emails, links, and names as plain text within a normal sentence."
)

# ---- Simple conversation memory (just a growing list of messages) ----
conversation_history = []

def ask_agent(question: str) -> str:
    conversation_history.append({"role": "user", "content": question})
    result = agent.invoke({"messages": conversation_history})
    ai_message = result["messages"][-1]
    conversation_history.append({"role": "assistant", "content": ai_message.content})
    return ai_message.content


if __name__ == "__main__":
    print("Agent ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        answer = ask_agent(user_input)
        print(f"\nAgent: {answer}\n")