from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits a list of LangChain Document objects into smaller chunks
    using RecursiveCharacterTextSplitter.
    
    Args:
        documents (list): A list of LangChain Document objects.
        chunk_size (int): Maximum character length of each chunk.
        chunk_overlap (int): Character overlap between adjacent chunks.
        
    Returns:
        list: A list of split LangChain Document objects.
    """
    if not documents:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    return chunks
