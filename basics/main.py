import os
from importlib.metadata import version
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    # Using the auto-updating alias to avoid deprecation errors
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()