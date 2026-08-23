import os
import tempfile
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv

load_dotenv()


# 1. Alternative for TextLoader (Native Python)
def load_text_file():
    print("\n--- Running Text Loader Alternative ---")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing.")
        temp_file_path = Path(temp_file.name)

    try:
        with open(temp_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        doc = Document(
            page_content=text, 
            metadata={"source": str(temp_file_path)}
        )
        
        documents = [doc]
        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")
    finally:
        os.remove(temp_file_path)


# 2. Alternative for WebBaseLoader (Requests + BeautifulSoup)
def load_web_page(url="https://example.com"):
    print(f"\n--- Running Web Loader Alternative for {url} ---")
    
    # Fetch the HTML
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the text out of the HTML
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    
    # Wrap in a LangChain Document
    doc = Document(
        page_content=text, 
        metadata={"source": url}
    )
    
    documents = [doc]
    print(f"Loaded {len(documents)} web document(s)")
    print(f"Content preview: {documents[0].page_content[:100]}...")
    print(f"Metadata: {documents[0].metadata}")


# 3. Alternative for DirectoryLoader (Native Python pathlib)
def load_directory(directory_path="."):
    print(f"\n--- Running Directory Loader Alternative for '{directory_path}' ---")
    documents = []
    
    # rglob("*.txt") recursively finds all .txt files in the directory
    for file_path in Path(directory_path).rglob("*.txt"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                doc = Document(
                    page_content=text, 
                    metadata={"source": str(file_path)}
                )
                documents.append(doc)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                
    print(f"Loaded {len(documents)} document(s) from directory")
    if documents:
        print(f"Content preview (first doc): {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")


# 4. Alternative for PyPDFLoader (langchain-unstructured)
def load_pdf_file(pdf_path="sample.pdf"):
    print(f"\n--- Running PDF Loader Alternative for '{pdf_path}' ---")
    
    if not os.path.exists(pdf_path):
        print(f"File '{pdf_path}' not found. Please place a valid PDF here to test.")
        return
        
    # UnstructuredLoader handles the complex parsing and chunking natively
    loader = UnstructuredLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s) from PDF")
    if documents:
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")


if __name__ == "__main__":
    # Test Text
    load_text_file()
    
    # Test Web
    load_web_page("https://example.com")
    
    # Test Directory (Searches current directory for .txt files)
    load_directory(".")
    
    # Test PDF (You will need a file named 'sample.pdf' in your folder for this to work)
    # If the file doesn't exist, it will safely print a warning and skip.
    load_pdf_file("docs/langchain_3_page_revision.pdf")