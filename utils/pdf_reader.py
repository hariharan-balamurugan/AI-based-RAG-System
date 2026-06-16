import os
from pypdf import PdfReader
from langchain_core.documents import Document

def extract_documents_from_pdf(pdf_file, filename=None):
    """
    Extracts text page-by-page from a PDF file (path or file-like object)
    and returns a list of LangChain Document objects with page-level metadata.
    
    Args:
        pdf_file: File path (str) or a file-like object (e.g. BytesIO, Streamlit UploadedFile)
        filename: Optional name of the file for metadata. Defaults to file name/path.
        
    Returns:
        List[Document]: List of LangChain Document objects.
    """
    if filename is None:
        if isinstance(pdf_file, str):
            filename = os.path.basename(pdf_file)
        else:
            filename = getattr(pdf_file, "name", "Uploaded PDF")

    try:
        reader = PdfReader(pdf_file)
        documents = []
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                # Store text page-by-page with source metadata (1-indexed pages)
                doc = Document(
                    page_content=page_text,
                    metadata={
                        "source": filename,
                        "page": i + 1
                    }
                )
                documents.append(doc)
                
        return documents
    except Exception as e:
        raise ValueError(f"Failed to read PDF file '{filename}': {str(e)}")
