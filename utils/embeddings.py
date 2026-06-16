from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes and returns a Sentence-Transformer embedding model using
    LangChain's HuggingFaceEmbeddings wrapper.
    
    Args:
        model_name (str): Hugging Face model identifier.
        
    Returns:
        HuggingFaceEmbeddings: The initialized embedding model object.
    """
    try:
        # Default to CPU execution for portable, serverless, or local use
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to load sentence-transformers embeddings: {str(e)}")
