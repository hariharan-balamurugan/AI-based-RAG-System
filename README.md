# PlacementPrep AI: AI-Powered Placement Preparation Assistant using RAG

PlacementPrep AI is a Retrieval-Augmented Generation (RAG) platform designed to help students prepare for campus placements, technical interviews, and core Computer Science subjects. By indexing study materials (PDFs) locally using a FAISS vector database and leveraging the Grok (xAI) API, the assistant generates highly contextual answers, custom aptitude quizzes, resume evaluations, and company-specific preparation guides.

---

## 🏗️ Architecture Diagram

The diagram below visualizes the data flow for both **Document Ingestion** (indexing) and **Retrieval-Augmented Generation** (query answering):

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion Pipeline"]
        A[Upload PDF Materials] --> B[PyPDF Reader]
        B -->|Extract Page-Level Text| C[RecursiveCharacterTextSplitter]
        C -->|1000-char Overlapping Chunks| D[Sentence Transformers: all-MiniLM-L6-v2]
        D -->|Vector Embeddings| E[(FAISS Vector Store)]
    end

    subgraph Generation["2. RAG Generation Pipeline"]
        F[Student Query] --> G[Sentence Transformers]
        G -->|Query Embedding| H[FAISS Similarity Search]
        E -->|Retrieve Top K Chunks| H
        H -->|Relevant Chunks + Page Metadata| I[Context Compilation]
        I -->|Structured Context + System Prompt| J[Grok (xAI) API]
        J -->|Generate Response| K[Chatbot Answer & Source Badges]
        F --> K
    end
```

---

## 🌟 Key Features

1. **💬 Contextual Placement Chatbot (RAG)**: Chat with study resources (Java, DBMS, OS, Computer Networks, Quantitative Aptitude). Every response shows clickable source cards detailing the filename and page numbers where the information was retrieved.
2. **📄 Resume Skill Analyzer**: Upload a resume in PDF format to get an instant profile match score, list of recognized skills, gap analysis (missing skills for software engineering roles), custom preparation roadmap, and 5 tailored interview questions with answers.
3. **🏢 Company Prep Mode**: Provide a target company name (e.g. TCS, Infosys, Google, Amazon) and upload past experience transcripts to generate a step-by-step interview round breakdown, target core topics, and 5 typical company-specific interview questions.
4. **✏️ Interactive Aptitude Practice**: Generates multiple-choice aptitude tests (Quantitative, Logical, CS Core) directly from uploaded documents, calculates real-time scores, and provides detailed step-by-step explanations for each question.

---

## 📂 Folder Structure & File Explanations

Here is the modular structure of the repository, followed by explanations for each file:

```
placement-rag/
│
├── app.py                     # Main Streamlit Web Application entrypoint
├── requirements.txt           # Python dependency specifications
├── .env                       # Environment variables (API Key templates)
│
├── data/                      # Folder to place default prep PDFs (e.g. Java.pdf, aptitude.pdf)
├── vector_store/              # Directory where local FAISS vector indices are stored
│
└── utils/                     # Modular helper functions directory
    ├── pdf_reader.py          # Extracts text page-by-page from PDF files into LangChain Documents
    ├── chunking.py            # Splits documents into smaller text chunks with metadata
    ├── embeddings.py          # Loads Sentence Transformers models locally (all-MiniLM-L6-v2)
    ├── retriever.py           # Handles FAISS index creation, local saving, loading, and queries
    └── grok_client.py         # Manages prompt compilation and API interactions with Grok (xAI)
```

### Beginner-Friendly Code Descriptions

*   **`app.py`**: Runs the Streamlit web server. Handles page layouts, sidebar inputs, custom glassmorphism styles, and manages active state history (chat conversations, quiz scores, active files).
*   **`requirements.txt`**: Declares package dependencies (Streamlit, LangChain, FAISS, Sentence Transformers, PyPDF, OpenAI SDK) required by pip.
*   **`utils/pdf_reader.py`**: Opens PDF documents and parses text page-by-page. For each page, it packs the text into a LangChain `Document` class and embeds metadata specifying the file name and page number.
*   **`utils/chunking.py`**: Takes full pages and divides them into smaller parts (chunks of 1000 characters). This ensures that search retrieval matches specific paragraphs instead of pulling entire pages, saving API token costs.
*   **`utils/embeddings.py`**: Loads the local SentenceTransformer embedding model. It turns text chunks into mathematical vectors (numbers representing meaning), enabling the database to search text by similarity rather than exact keyword match.
*   **`utils/retriever.py`**: Standardizes database controls. Saves or loads the FAISS index locally to/from `/vector_store`, appends new chunks incrementally, and finds top-matches for a student's question.
*   **`utils/grok_client.py`**: Packages API requests to Grok. It sets special instructions (System Prompts) explaining to Grok how to behave (e.g., as an HR recruiter or an aptitude trainer) and feeds in the retrieved document context.

---

## 🛠️ Setup & Installation Instructions

Follow these steps to run the application locally on your Windows machine:

### 1. Prerequisite
Ensure you have Python 3.10+ installed. You can verify this by running:
```powershell
python --version
```

### 2. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to manage dependencies cleanly:
```powershell
# Create environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install all package requirements listed in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root folder (or use the existing template) and add your Grok API Key:
```env
XAI_API_KEY=your_actual_grok_api_key_here
```
> 💡 *Note: You can also enter the API key directly in the Streamlit Sidebar UI during testing.*

### 5. Running the Application
Launch the Streamlit app by executing:
```powershell
python -m streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

---

## 📄 Resume Project Description

If you built this project and want to showcase it on your resume, here is a professional, recruiter-friendly description you can use:

> **AI-Powered Placement Preparation Assistant | Python, Streamlit, LangChain, FAISS, Grok (xAI) API, HuggingFace**
> *   Designed and implemented a Retrieval-Augmented Generation (RAG) system to index academic and aptitude preparation PDF files, improving prep efficiency.
> *   Utilized **LangChain** and **Sentence Transformers (`all-MiniLM-L6-v2`)** to generate dense semantic vector embeddings, storing them in a local **FAISS** index for millisecond-latency similarity search.
> *   Integrated **Grok (xAI) API** using OpenAI-compatible SDK clients and custom system instructions to construct multi-turn technical chatbot assistance and dynamic aptitude quiz generation.
> *   Developed a **Resume Analyzer** component that extracts resume content page-by-page, performs automated skill-mapping, lists candidate gaps, and drafts customized interview questions with model answers.
> *   Built a responsive, glassmorphism-themed frontend using **Streamlit** incorporating session-state tracking, custom CSS styling, and document-level citation references.

---

## 💬 Interview Questions & Answers (Project-Specific)

If an interviewer asks you about this project, here are typical questions you might face, along with suggested answers:

### Q1: What is RAG, and why did you use it instead of just asking Grok questions directly?
*   **Answer**: RAG stands for Retrieval-Augmented Generation. If we ask Grok questions directly, it relies on its general pre-trained knowledge, which might miss company-specific guidelines, custom syllabus nodes, or student notes. Additionally, LLMs are prone to hallucinating details. By using RAG, we search a local database (FAISS) for precise reference passages first, feed them into Grok as ground-truth context, and instruct Grok to answer using only that context. This reduces hallucinations and guarantees answers align with our actual files.

### Q2: Why did you choose FAISS over cloud vector stores like Pinecone or Milvus?
*   **Answer**: FAISS (Facebook AI Similarity Search) is a lightweight, open-source library optimized for in-memory vector calculations. Since this placement assistant is designed to run locally on consumer hardware, FAISS allows us to save and load index files (`.faiss` and `.pkl`) directly to disk in the project directory without requiring cloud registration, API tokens, network latency, or external database charges.

### Q3: Why did you use `all-MiniLM-L6-v2` for embeddings instead of OpenAI or Google Embeddings APIs?
*   **Answer**: `all-MiniLM-L6-v2` by Sentence-Transformers is a highly popular, compact model (only ~120MB) that runs entirely on local CPUs. It provides 384-dimensional vector embeddings with high semantic quality. Using it makes the system self-contained, eliminates network overhead for generating index embeddings, and keeps the project free of API cost charges during document ingestion.

### Q4: How did you implement xAI's Grok API in this project?
*   **Answer**: xAI designed their API to be fully compatible with standard OpenAI SDK structures. We used the standard `openai` library client, specifying Grok's base URL (`https://api.x.ai/v1`) and passing the API key. We routed the requests to the `grok-beta` model. This allowed us to easily plug Grok into our application using standard Chat Completion endpoints.

### Q5: How did you ensure that the application displays citations for retrieved chunks?
*   **Answer**: In `utils/pdf_reader.py`, when text is extracted, page metadata (`source` and `page`) is added to each LangChain `Document`. During splitting, `RecursiveCharacterTextSplitter` propagates this metadata to each split chunk. When a similarity search is executed, FAISS returns the matching chunks with their metadata intact. We store this metadata in the Streamlit session state chat history and display it as clickable reference badges below each chatbot response.
