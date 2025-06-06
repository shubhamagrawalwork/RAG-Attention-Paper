import os
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain import hub


# --- Load Environment Variables ---
load_dotenv()


# --- Configuration ---
pdf_path = "attention.pdf" 
mongo_uri = os.getenv("MONGO_URI")
google_api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY") # For LangSmith tracing
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_project = os.getenv("LANGCHAIN_PROJECT")


# Set environment variables for LangChain/Google SDKs
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
os.environ["LANGCHAIN_PROJECT"] = langchain_project

db_name = "RegPractise"
collection_name = "attention"
vector_index_name = "vector_index" # Your MongoDB Atlas vector index name

if not all([mongo_uri, google_api_key]):
    raise ValueError("Please ensure MONGO_URI and GOOGLE_API_KEY are set in your .env file.")


# --- 1. Initial Setup and Data Ingestion (LangChain Way) ---
print("\n--- Starting Data Ingestion ---")


# Load Documents
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} pages from PDF")


# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Connect to MongoDB Atlas
client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]


# Optional: Clear existing documents if you want to re-ingest every time
collection.delete_many({})


# Ingest Documents into MongoDB Atlas Vector Search
# This will check if documents already exist based on content and metadata (if applicable)
# or just insert if the collection is empty.
print("Ingesting documents into MongoDB Atlas Vector Search...")
vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=documents,
    embedding=embeddings,
    collection=collection,
    index_name=vector_index_name
)
print("All documents embedded and stored in MongoDB Atlas via LangChain!")




# --- 2. Retrieval Augmented Generation (RAG) Pipeline ---
print("\n--- Building the RAG Pipeline ---")


# Define the Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Get top 5 relevant chunks
print("Retriever configured.")


# Initialize the Large Language Model (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)


# Define the Prompt Template - FROM LANGCHAIN HUB
prompt = hub.pull("rlm/rag-prompt")
import pprint
pprint.pprint(prompt.messages)


# Build the RAG Chain using LangChain Expression Language (LCEL)
def format_docs(docs):
    """Helper function to concatenate the page_content of retrieved documents."""
    # Ensure it works whether docs are (Document, score) tuples or just Document objects
    if docs and isinstance(docs[0], tuple) and isinstance(docs[0][0], Any):
         return "\n\n".join(doc[0].page_content for doc in docs)
    else:
         return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("RAG chain built successfully!")




# ---3. Testing Your RAG Pipeline ---


if __name__ == "__main__":
    print("\n--- Testing the RAG Pipeline ---")

    # Example 1
    user_query_1 = "What is the main innovation proposed in the paper?"
    print(f"\nQuestion 1: {user_query_1}")
    answer_1 = rag_chain.invoke(user_query_1)
    print(f"Answer 1: {answer_1}")

    # Example 2
    user_query_2 = "How do they handle position information without recurrence?"
    print(f"\nQuestion 2: {user_query_2}")
    answer_2 = rag_chain.invoke(user_query_2)
    print(f"Answer 2: {answer_2}")

    # Example 3 (Outside of context)
    user_query_3 = "What is the capital of France?"
    print(f"\nQuestion 3: {user_query_3}")
    answer_3 = rag_chain.invoke(user_query_3)
    print(f"Answer 3: {answer_3}") # Should indicate it can't find the answer in the context
