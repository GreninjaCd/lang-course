import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
LOCAL_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)


def normalize_database_url(url: str) -> str:
    """Ensure Supabase URLs include the required SSL flag."""
    if not url:
        return url
    if ".supabase.co" in url and "sslmode=" not in url:
        return f"{url}?sslmode=require"
    return url


def get_connection_attempts():
    attempts = []
    if SUPABASE_URL:
        attempts.append(("Supabase", normalize_database_url(SUPABASE_URL)))
    if LOCAL_DATABASE_URL:
        attempts.append(("Local Postgres", LOCAL_DATABASE_URL))
    if not attempts:
        attempts.append(("Local Postgres", "postgresql://postgres:postgres@localhost:5432/postgres"))
    return attempts


def connect_to_supabase():
    """Connect to Supabase pgvector or fallback to local Postgres."""
    last_error = None
    for label, connection_url in get_connection_attempts():
        try:
            print(f"\nTrying {label} connection...")
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
            vectorstore = PGVector(
                embeddings=embeddings,
                collection_name="production_docs",
                connection=connection_url,
                use_jsonb=True,
            )
            print(f"Connected to {label} successfully.")
            return vectorstore
        except Exception as exc:
            last_error = exc
            print(f"{label} connection failed: {exc}")

    raise RuntimeError(
        "Could not connect to Supabase or local Postgres. "
        "Check that the database is running, the connection string is correct, "
        "and your Supabase project allows your IP address."
    ) from last_error


def verify_connection(vectorstore):
    """Verify the connection works."""
    from langchain_core.documents import Document

    test_doc = Document(
        page_content="This is a test doc to verify the vector database",
        metadata={"test": True},
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Added test docs: {ids[0]}")

        results = vectorstore.similarity_search("test document")
        if results:
            print(f"Search works: {results[0].page_content}")

        vectorstore.delete(ids)
        print("cleanup complete")
        return True

    except Exception as exc:
        print(f"Verification failed: {exc}")
        return False


def main():
    print("=" * 60)
    print("Supabase pgvector connection test")
    print("=" * 60)

    try:
        vectorstore = connect_to_supabase()
        status = verify_connection(vectorstore)

        if status:
            print("\nConnection verification successful.")
        else:
            print("\nConnection verification failed.")
    except Exception as exc:
        print("\nDatabase connection could not be established.")
        print(f"Reason: {exc}")
        print("\nChecklist:")
        print("1. Confirm the Supabase project is active.")
        print("2. Add your IP to Supabase Network settings.")
        print("3. Verify the DATABASE_URL or SUPABASE_DATABASE_URL is correct.")
        print("4. If using local Postgres, make sure postgres is running on localhost:5432.")


if __name__ == "__main__":
    main()