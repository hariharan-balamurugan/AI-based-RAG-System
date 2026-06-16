import os
from langchain_community.vectorstores import FAISS

VECTOR_DB_DIR = "vector_store"

def create_vector_store(documents, embeddings_model):
    """
    Creates a new FAISS vector store from a list of LangChain Document objects.
    
    Args:
        documents (list): List of Document objects.
        embeddings_model: Embedded model interface.
        
    Returns:
        FAISS: Created FAISS vector store object, or None if input list is empty.
    """
    if not documents:
        return None
    return FAISS.from_documents(documents, embeddings_model)

def save_vector_store(vector_store, path=VECTOR_DB_DIR):
    """
    Saves the FAISS index and local metadata to a directory.
    
    Args:
        vector_store (FAISS): FAISS index object to save.
        path (str): Target directory.
    """
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)

def load_vector_store(path=VECTOR_DB_DIR, embeddings_model=None):
    """
    Loads an existing FAISS index from the local directory.
    
    Args:
        path (str): Directory containing index.faiss and index.pkl.
        embeddings_model: The embedding model utilized when building the database.
        
    Returns:
        FAISS: The loaded FAISS vector store, or None if index files do not exist.
    """
    if not os.path.exists(path) or not os.path.exists(os.path.join(path, "index.faiss")):
        return None
    try:
        # allow_dangerous_deserialization is required for loading local pickle files containing metadata
        return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Warning: Failed to load existing FAISS vector store: {str(e)}")
        return None

def add_to_vector_store(vector_store, documents, embeddings_model, save_path=VECTOR_DB_DIR):
    """
    Appends new documents to an existing vector store and updates the stored files.
    If no vector store exists, it creates one.
    
    Args:
        vector_store (FAISS): Existing FAISS instance (can be None).
        documents (list): List of new Document objects.
        embeddings_model: Embeddings model.
        save_path (str): Path to save directory.
        
    Returns:
        FAISS: The updated FAISS vector store instance.
    """
    if not documents:
        return vector_store
        
    if vector_store is None:
        vector_store = create_vector_store(documents, embeddings_model)
    else:
        vector_store.add_documents(documents)
        
    if vector_store is not None:
        save_vector_store(vector_store, save_path)
        
    return vector_store

def retrieve_relevant_documents(query, vector_store, k=5):
    """
    Performs similarity search and retrieves the top-K relevant documents.
    
    Args:
        query (str): The search query.
        vector_store (FAISS): The FAISS instance.
        k (int): Number of documents to retrieve.
        
    Returns:
        list: List of retrieved LangChain Document objects.
    """
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)
