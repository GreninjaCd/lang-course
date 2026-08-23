from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

def create_kb():
    """Create a vector store from knowledfe base."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content = KNOWLEDGE_BASE, metadata = {"source": "langchain_knowledge_base.md"})

    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents = chunks, embedding = embeddings_model, persist_directory = tempfile.mkdtemp(),
    )

    return vector_store

def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(seracg_type = "similarity", search_kwargs = {"k": 2})
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:

Make sure to answer in a concise manner and keep it at few lines,
and if you don't know the answer, just say "I don't know."""
    )

    # Format retrived docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    demo_basic_rag()
