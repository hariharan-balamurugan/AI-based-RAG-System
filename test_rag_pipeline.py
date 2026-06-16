import os
import sys

# Ensure current directory is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pdf_reader import extract_documents_from_pdf
from utils.chunking import split_documents
from utils.embeddings import get_embeddings_model
from utils.retriever import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    retrieve_relevant_documents
)

def test_pipeline():
    print("--- Starting RAG Pipeline Integration Test ---")
    
    # 1. Test PDF Reading
    pdf_path = os.path.join("data", "Java.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: Sample PDF not found at {pdf_path}. Skipping PDF reading test.")
        return False
        
    print(f"Loading sample PDF: {pdf_path}...")
    try:
        documents = extract_documents_from_pdf(pdf_path)
        print(f"Successfully read PDF. Extracted {len(documents)} pages.")
        
        if len(documents) == 0:
            print("Error: No pages extracted.")
            return False
            
        print(f"Page 1 metadata: {documents[0].metadata}")
        print(f"Page 1 text snippet: {documents[0].page_content[:150]}...\n")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return False
    
    # 2. Test Chunking
    print("Splitting pages into chunks...")
    try:
        # Test with the first 3 pages to make it quick
        chunks = split_documents(documents[:3])
        print(f"Created {len(chunks)} chunks.")
        if len(chunks) == 0:
            print("Error: Chunking returned 0 chunks.")
            return False
        print(f"Chunk 1 length: {len(chunks[0].page_content)} characters")
        print(f"Chunk 1 metadata: {chunks[0].metadata}\n")
    except Exception as e:
        print(f"Error during chunking: {e}")
        return False
    
    # 3. Test Embedding Initialization
    print("Loading SentenceTransformers embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    try:
        embeddings = get_embeddings_model()
        print("Embeddings loaded successfully.\n")
    except Exception as e:
        print(f"Error loading embeddings model: {e}")
        return False
        
    # 4. Test FAISS Storage & Retrieval
    print("Building local FAISS index...")
    try:
        vector_store = create_vector_store(chunks, embeddings)
        print("FAISS index built successfully.")
        
        save_path = "test_vector_store"
        print(f"Saving vector store to temporary directory '{save_path}'...")
        save_vector_store(vector_store, save_path)
        
        print("Loading vector store from disk...")
        loaded_store = load_vector_store(save_path, embeddings)
        if loaded_store is None:
            print("Error: Loaded vector store is None.")
            return False
        print("Loaded vector store successfully.")
        
        # Test Similarity Search
        query = "Java garbage collection or memory"
        print(f"Searching index for query: '{query}'...")
        results = retrieve_relevant_documents(query, loaded_store, k=2)
        print(f"Found {len(results)} matches:")
        for idx, doc in enumerate(results):
            print(f"Match {idx+1}: File: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
            # Encode/decode to remove non-ASCII characters that break Windows console printing
            safe_text = doc.page_content[:200].encode('ascii', errors='ignore').decode('ascii')
            print(f"Text snippet: {safe_text}...\n")
            
        # Clean up test index folder
        import shutil
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            print(f"Cleaned up temporary index files at '{save_path}'.")
            
        print("--- RAG Backend Integration Test Passed Successfully! ---")
        return True
    except Exception as e:
        print(f"Error during FAISS build, load or retrieval: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
