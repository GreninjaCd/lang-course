import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# 1. Initialize the Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    google_api_key=api_key
)

# 2. Initialize the Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key
)

# --- Chat Model Request ---
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = llm.invoke(messages)
print("Chat Response:")
print(response.text)

# --- Embedding Request ---
text_to_embed = "What is the capital of France?"

# Generate numerical vector representation
vector = embeddings.embed_query(text_to_embed)

print(vector)
print("\nEmbedding Response:")
print(f"Total Vector Length: {len(vector)}")
print(f"Sample Dimensions: {vector[:5]}...")