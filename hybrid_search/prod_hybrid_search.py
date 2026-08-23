import warnings
# Suppress the LangChain community deprecation warnings for cleaner console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview") # Updated to latest standard model

documents = [
    Document(
        page_content='Product SKU-7742X is our flagship router. It supports '
                     'gigabit speeds and advanced QoS features.',
        metadata={'type': 'product'}
    ),
    Document(
        page_content='For network connectivity issues, first check the '
                     'ethernet cable and router status lights.',
        metadata={'type': 'troubleshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server '
                     'rejected the connection. Check firewall settings.',
        metadata={'type': 'error'}
    ),
    Document(
        page_content='The authentication process requires valid credentials. '
                     'Use OAuth2 for secure API access.',
        metadata={'type': 'auth'}
    ),
    Document(
        page_content='Router configuration guide: Access the admin panel '
                     'at 192.168.1.1 to modify settings.',
        metadata={'type': 'config'}
    ),
    Document(
        page_content='WCAG 2.1 compliance requires all images to have '
                     'alt text and sufficient color contrast.',
        metadata={'type': 'compliance'}
    ),
]

print(f'Loaded {len(documents)} documents')

# 1. Vector Retriever Setup
vectorstore = Chroma.from_documents(
    documents, embeddings, collection_name='hybrid_test'
)
vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
print('Vector retriever ready')

# 2. BM25 Retriever Setup
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3
print('BM25 retriever ready')

# 3. Custom Ensemble Retriever Function
def custom_ensemble_search(query, retrievers, weights, top_k=3, c=60):
    """
    Combines results from multiple retrievers using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    doc_map = {}

    for retriever, weight in zip(retrievers, weights):
        # Get results from the current retriever
        docs = retriever.invoke(query)
        
        for rank, doc in enumerate(docs):
            # Use page_content as a unique identifier (or a hash if content is huge)
            doc_id = doc.page_content
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                doc_map[doc_id] = doc
                
            # RRF calculation: weight / (c + rank)
            # rank is 0-indexed in Python, so we add 1 for the mathematical formula
            rrf_scores[doc_id] += weight / (c + rank + 1)

    # Sort documents by their accumulated RRF score in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top_k Document objects
    return [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_k]]

print('Custom Hybrid ensembler ready')

# 4. Testing utility
def test_query(query, name, retriever=None, is_hybrid=False):
    '''Test a query and show results'''
    if is_hybrid:
        # Pass the retrievers directly to our custom function
        results = custom_ensemble_search(
            query=query, 
            retrievers=[bm25_retriever, vector_retriever], 
            weights=[0.5, 0.5]
        )
    else:
        results = retriever.invoke(query)
        
    print(f'\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        # Cleaned up string slicing formatting
        preview = doc.page_content[:80].strip() + '...'
        print(f' {i + 1}. {preview}')
    return results

test_queries = [
    'SKU-7742X specifications',
    'E_CONN_REFUSED error',
    'How do I authenticate?',
    'WCAG compliance',
    'router configuration',
]

for query in test_queries:
    print('=' * 60)
    vector_results = test_query(query, 'VECTOR', retriever=vector_retriever)
    bm25_results = test_query(query, 'BM25', retriever=bm25_retriever)
    hybrid_results = test_query(query, 'HYBRID', is_hybrid=True)