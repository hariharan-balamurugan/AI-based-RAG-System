import os
import streamlit as st
from dotenv import load_dotenv

# Import utilities
from utils.pdf_reader import extract_documents_from_pdf
from utils.chunking import split_documents
from utils.embeddings import get_embeddings_model
from utils.retriever import (
    load_vector_store,
    add_to_vector_store,
    retrieve_relevant_documents
)
from utils.llm_client import (
    generate_rag_answer,
    analyze_resume,
    generate_company_prep,
    generate_aptitude_quiz
)

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="PlacementPrep AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the SentenceTransformer model to avoid reloading on every refresh
@st.cache_resource
def load_cached_embeddings():
    return get_embeddings_model()

# Custom CSS for modern glassmorphism UI
def load_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Typography adjustments */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }

        /* Title gradient animation styling */
        .main-title {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            text-align: center;
            font-size: 2.8rem;
            margin-bottom: 0.1rem;
            letter-spacing: -0.05em;
        }
        
        .sub-title {
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 1.8rem;
            font-weight: 400;
        }

        /* Premium Glass Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.08);
        }
        
        /* Card headers */
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 10px;
        }
        
        .card-content {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        /* Tags and badges */
        .source-badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #c7d2fe;
            margin-right: 5px;
            margin-top: 5px;
        }
        
        .source-badge-company {
            display: inline-block;
            background: rgba(236, 72, 153, 0.15);
            border: 1px solid rgba(236, 72, 153, 0.3);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #fbcfe8;
            margin-right: 5px;
            margin-top: 5px;
        }

        /* Sidebar Styling overrides */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        
        /* Floating background glow effect */
        .glow-effect-1 {
            position: absolute;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, rgba(0,0,0,0) 70%);
            top: -50px;
            left: -50px;
            z-index: -2;
            pointer-events: none;
        }
        
        .glow-effect-2 {
            position: absolute;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.08) 0%, rgba(0,0,0,0) 70%);
            bottom: -50px;
            right: -50px;
            z-index: -2;
            pointer-events: none;
        }
        </style>
        <div class="glow-effect-1"></div>
        <div class="glow-effect-2"></div>
    """, unsafe_allow_html=True)

# Initialize Session States
embeddings = load_cached_embeddings()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store(embeddings_model=embeddings)
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None
if "company_analysis" not in st.session_state:
    st.session_state.company_analysis = None
if "aptitude_quiz" not in st.session_state:
    st.session_state.aptitude_quiz = None
if "aptitude_answers" not in st.session_state:
    st.session_state.aptitude_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

# Helper: Read index names
def get_indexed_files():
    if st.session_state.vector_store is None:
        return []
    try:
        docstore = st.session_state.vector_store.docstore._dict
        sources = set()
        for doc in docstore.values():
            sources.add(doc.metadata.get("source", "Unknown"))
        return sorted(list(sources))
    except Exception:
        return []

# SIDEBAR CONFIGURATION
st.sidebar.markdown("<h2 style='text-align: center; font-family: Outfit;'>🎓 PrepPanel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# API Key Config
st.sidebar.subheader("🔑 API Configuration")

# Try reading from Streamlit Secrets first (for deployment), fallback to local .env
try:
    api_key = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
except Exception:
    api_key = os.getenv("HF_TOKEN", "")

if not api_key:
    st.sidebar.warning("⚠️ HF_TOKEN is missing. Please add it to your .env or st.secrets. Generation features are disabled.")
else:
    st.sidebar.success("✅ Hugging Face API connected.")

st.sidebar.markdown("---")

# Navigation Menu
st.sidebar.subheader("🧭 Navigation")
menu = ["💬 Placement Chatbot", "📄 Resume Analyzer", "🏢 Company Prep Mode", "✏️ Aptitude Practice"]
choice = st.sidebar.radio("Go to Section", menu)

st.sidebar.markdown("---")

# Documents Manager
st.sidebar.subheader("📚 Placement Resources")
uploaded_files = st.sidebar.file_uploader(
    "Upload Prep Materials (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload Aptitude, DBMS, OS, Java, CN, or Company PDF materials."
)

if uploaded_files:
    if st.sidebar.button("⚡ Index Uploaded Files", use_container_width=True):
        with st.spinner("Analyzing & embedding documents..."):
            all_documents = []
            for f in uploaded_files:
                try:
                    docs = extract_documents_from_pdf(f)
                    all_documents.extend(docs)
                except Exception as e:
                    st.sidebar.error(f"Error reading {f.name}: {e}")
            
            if all_documents:
                chunks = split_documents(all_documents)
                st.session_state.vector_store = add_to_vector_store(
                    st.session_state.vector_store,
                    chunks,
                    embeddings
                )
                st.sidebar.success(f"Indexed {len(uploaded_files)} files into {len(chunks)} chunks!")
            else:
                st.sidebar.error("Could not extract any content from the PDFs.")

# Preloaded Directory indexing
preloaded_dir = "data"
if os.path.exists(preloaded_dir):
    pdf_files = [f for f in os.listdir(preloaded_dir) if f.endswith(".pdf")]
    if pdf_files:
        if st.sidebar.button("📂 Index Sample PDFs (Java/Aptitude)", use_container_width=True):
            with st.spinner("Indexing sample PDFs from 'data/'..."):
                all_documents = []
                for file_name in pdf_files:
                    file_path = os.path.join(preloaded_dir, file_name)
                    try:
                        docs = extract_documents_from_pdf(file_path, filename=file_name)
                        all_documents.extend(docs)
                    except Exception as e:
                        st.sidebar.error(f"Error loading {file_name}: {e}")
                if all_documents:
                    chunks = split_documents(all_documents)
                    st.session_state.vector_store = add_to_vector_store(
                        st.session_state.vector_store,
                        chunks,
                        embeddings
                    )
                    st.sidebar.success(f"Indexed preloaded files: {', '.join(pdf_files)}!")

# Show list of loaded documents in sidebar
indexed_files = get_indexed_files()
if indexed_files:
    with st.sidebar.expander(f"📁 Indexed Files ({len(indexed_files)})", expanded=False):
        for index_file in indexed_files:
            st.markdown(f"- `{index_file}`")
else:
    st.sidebar.info("No documents indexed yet. Upload PDFs or load sample files.")

# APP INTERFACE
load_custom_css()

# Render Title
st.markdown("<h1 class='main-title'>PlacementPrep AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Your AI-Powered Retrieval-Augmented Placement Coach</p>", unsafe_allow_html=True)

# ----------------- SECTION 1: Placement Chatbot (RAG) -----------------
if choice == "💬 Placement Chatbot":
    st.markdown("### 💬 Chatbot Prep Assistant")
    st.markdown("Ask anything about Quantitative Aptitude, DBMS, Operating Systems, Computer Networks, Java, or specific placement prep queries. The bot will search your indexed documents to form context-driven responses.")

    # Suggestions for quick prompts
    st.markdown("**💡 Quick Prep Queries:**")
    cols = st.columns(4)
    suggested_prompts = [
        "Explain method overloading vs overriding in Java.",
        "What are normal forms in DBMS? Explain 3NF.",
        "Explain the working of TCP three-way handshake.",
        "How do you solve Speed, Distance & Time problems?"
    ]
    
    # Store clicked prompt
    clicked_prompt = None
    for idx, prompt_text in enumerate(suggested_prompts):
        if cols[idx].button(prompt_text, key=f"sug_{idx}", use_container_width=True):
            clicked_prompt = prompt_text

    # Show Chat Messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 Retrieved Document References", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="glass-card" style="padding: 10px; margin-bottom: 8px;">
                            <span class="source-badge">📄 {src['file']} (Page {src['page']})</span>
                            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px; font-style: italic;">
                                "... {src['text']} ..."
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # User Input
    chat_input = st.chat_input("Ask a technical or aptitude preparation question...")
    
    # Override chat_input if suggestion was clicked
    user_query = chat_input if chat_input else clicked_prompt

    if user_query:
        # User input display
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Save to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Retrieval and response generation
        with st.chat_message("assistant"):
            if not api_key:
                st.error("Please supply a valid Hugging Face API Key (.env or st.secrets) to generate answers.")
            else:
                with st.spinner("Searching document context and drafting response..."):
                    # Check if vector store is loaded
                    retrieved_docs = []
                    sources_data = []
                    
                    if st.session_state.vector_store is not None:
                        retrieved_docs = retrieve_relevant_documents(user_query, st.session_state.vector_store, k=4)
                        for doc in retrieved_docs:
                            sources_data.append({
                                "file": doc.metadata.get("source", "Unknown"),
                                "page": doc.metadata.get("page", "?"),
                                "text": doc.page_content[:250].replace("\n", " ")
                            })
                    else:
                        st.info("ℹ️ No documents indexed. Bot will reply from general historical knowledge.")

                    answer = generate_rag_answer(user_query, retrieved_docs, api_key)
                    if answer.startswith("Error:"):
                        st.error(answer)
                    else:
                        st.markdown(answer)
                    
                    # Display Sources
                    if sources_data:
                        with st.expander("🔍 Retrieved Document References", expanded=False):
                            for src in sources_data:
                                st.markdown(f"""
                                <div class="glass-card" style="padding: 10px; margin-bottom: 8px;">
                                    <span class="source-badge">📄 {src['file']} (Page {src['page']})</span>
                                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px; font-style: italic;">
                                        "... {src['text']} ..."
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Save Assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_data
                    })

# ----------------- SECTION 2: Resume Analyzer -----------------
elif choice == "📄 Resume Analyzer":
    st.markdown("### 📄 Resume Skill Mapping & Interview Coach")
    st.markdown("Upload your Resume in PDF format. The assistant will rate your resume strength, map your skills, perform gap analysis, and generate customized interview questions with model answers.")
    
    resume_file = st.file_uploader("Upload Resume PDF File", type=["pdf"])
    
    if resume_file:
        if st.button("🚀 Analyze Resume", use_container_width=True):
            if not api_key:
                st.error("Provide a valid Hugging Face API Key first!")
            else:
                with st.spinner("Analyzing resume content... This may take a few seconds."):
                    try:
                        # Extract resume text
                        docs = extract_documents_from_pdf(resume_file)
                        resume_text = "\n".join([doc.page_content for doc in docs])
                        
                        if resume_text.strip():
                            analysis_result = analyze_resume(resume_text, api_key)
                            if analysis_result.startswith("Error:"):
                                st.error(analysis_result)
                            else:
                                st.session_state.resume_analysis = analysis_result
                        else:
                            st.error("Empty resume file text extracted.")
                    except Exception as e:
                        st.error(f"Error reading resume PDF: {e}")
                        
    if st.session_state.resume_analysis:
        st.success("Analysis Complete!")
        st.markdown("---")
        st.markdown(st.session_state.resume_analysis)
        
        # Clear button
        if st.button("🗑️ Clear Analysis"):
            st.session_state.resume_analysis = None
            st.rerun()

# ----------------- SECTION 3: Company Prep Mode -----------------
elif choice == "🏢 Company Prep Mode":
    st.markdown("### 🏢 Company Specific Interview Preparation")
    st.markdown("Generate placement guidelines, hiring structures, and technical coding questions tailored specifically for target companies. Optionally upload past papers or interview transcripts to build precise, custom guides.")
    
    col_c1, col_c2 = st.columns([2, 3])
    
    with col_c1:
        company_name = st.text_input("Enter Company Name", placeholder="e.g. TCS, Infosys, Amazon, Google")
        company_files = st.file_uploader("Upload Company Docs / Past Questions (Optional PDF)", type=["pdf"], accept_multiple_files=True)
    
    with col_c2:
        if st.button("🏭 Generate Company Prep Guide", use_container_width=True):
            if not company_name:
                st.error("Please enter a company name.")
            elif not api_key:
                st.error("Please insert a valid Hugging Face API Key.")
            else:
                with st.spinner(f"Compiling preparation strategy for {company_name}..."):
                    try:
                        company_docs_text = ""
                        if company_files:
                            all_comp_docs = []
                            for f in company_files:
                                all_comp_docs.extend(extract_documents_from_pdf(f))
                            company_docs_text = "\n".join([d.page_content for d in all_comp_docs])
                        
                        analysis_result = generate_company_prep(company_name, company_docs_text, api_key)
                        if analysis_result.startswith("Error:"):
                            st.error(analysis_result)
                        else:
                            st.session_state.company_analysis = analysis_result
                    except Exception as e:
                        st.error(f"Error compiling company materials: {e}")
                        
    if st.session_state.company_analysis:
        st.success(f"Tailored {company_name} Prep Guide ready!")
        st.markdown("---")
        st.markdown(st.session_state.company_analysis)
        
        if st.button("🗑️ Clear Prep Guide"):
            st.session_state.company_analysis = None
            st.rerun()

# ----------------- SECTION 4: Aptitude Practice Mode -----------------
elif choice == "✏️ Aptitude Practice":
    st.markdown("### ✏️ Interactive Aptitude Practice Quiz")
    st.markdown("Test your Quantitative Aptitude, Logical Reasoning, or core Computer Science concepts. The assistant extracts data from relevant RAG documents to build multiple-choice questions with step-by-step mathematical explanations.")
    
    col_q1, col_q2 = st.columns([2, 1])
    
    with col_q1:
        topic = st.selectbox(
            "Select Quiz Topic",
            ["Quantitative Aptitude (Percentages, Profit & Loss, Speed-Time)", 
             "Logical Reasoning (Syllogisms, Blood Relations, Coding-Decoding)", 
             "Core Computer Science (Java, DBMS, Operating Systems, Networks)"]
        )
    with col_q2:
        num_questions = st.slider("Number of Questions", min_value=3, max_value=10, value=5)
        
    if st.button("🎯 Generate Custom Quiz", use_container_width=True):
        if not api_key:
            st.error("Please insert a valid Hugging Face API key.")
        else:
            with st.spinner("Generating custom MCQ questions from documents..."):
                retrieved_context = []
                if st.session_state.vector_store is not None:
                    # Query the vector store for context on the selected topic to base questions on documents
                    retrieved_context = retrieve_relevant_documents(topic, st.session_state.vector_store, k=5)
                
                quiz_data = generate_aptitude_quiz(topic, retrieved_context, num_questions, api_key)
                
                # Store in session state
                st.session_state.aptitude_quiz = quiz_data
                st.session_state.aptitude_answers = {}
                st.session_state.quiz_submitted = False
                
    st.markdown("---")
    
    if st.session_state.aptitude_quiz:
        st.markdown(f"#### 📝 Practice Test: {topic}")
        
        # Display each question inside a form or list
        for idx, q in enumerate(st.session_state.aptitude_quiz):
            st.markdown(f"**Question {idx + 1}:** {q['question']}")
            
            # Map choices
            options = q["options"]
            
            # Use radio button, checking if answered
            default_index = None
            current_ans = st.session_state.aptitude_answers.get(q["id"])
            if current_ans in options:
                default_index = options.index(current_ans)
                
            selection = st.radio(
                f"Choose option for Question {idx + 1}:",
                options,
                index=default_index,
                key=f"q_radio_{q['id']}",
                label_visibility="collapsed"
            )
            
            # Update answers in session state
            st.session_state.aptitude_answers[q["id"]] = selection
            st.markdown("<br>", unsafe_allow_html=True)
            
        if not st.session_state.quiz_submitted:
            if st.button("📊 Submit Quiz", use_container_width=True):
                # Ensure all questions have a choice
                missing = [q['id'] for q in st.session_state.aptitude_quiz if not st.session_state.aptitude_answers.get(q['id'])]
                if missing:
                    st.warning("⚠️ Please answer all questions before submitting.")
                else:
                    st.session_state.quiz_submitted = True
                    st.rerun()
        else:
            # Score calculations
            score = 0
            total = len(st.session_state.aptitude_quiz)
            for q in st.session_state.aptitude_quiz:
                if st.session_state.aptitude_answers.get(q["id"]) == q["answer"]:
                    score += 1
                    
            st.balloons()
            
            # Show score
            percentage = (score / total) * 100
            if percentage >= 70:
                st.success(f"🎉 Excellent! Score: **{score}/{total}** ({percentage:.1f}%)")
            elif percentage >= 40:
                st.info(f"👍 Good Job! Score: **{score}/{total}** ({percentage:.1f}%)")
            else:
                st.warning(f"📚 Practice More! Score: **{score}/{total}** ({percentage:.1f}%)")
                
            st.markdown("### 🔍 Detailed Solutions")
            
            for idx, q in enumerate(st.session_state.aptitude_quiz):
                user_ans = st.session_state.aptitude_answers.get(q["id"])
                correct_ans = q["answer"]
                is_correct = user_ans == correct_ans
                
                # Output result
                status_icon = "✅ Correct" if is_correct else "❌ Incorrect"
                status_color = "green" if is_correct else "red"
                
                # Pre-format the explanation to avoid backslash inside the f-string (for Python < 3.12 compatibility)
                explanation_html = q['explanation'].replace('\n', '<br>')
                
                st.markdown(f"""
                <div class="glass-card" style="border-left: 5px solid {status_color};">
                    <strong>Question {idx+1}:</strong> {q['question']}<br>
                    <span style="color: {status_color}; font-weight: bold;">{status_icon}</span><br>
                    • Your Choice: <code>{user_ans}</code><br>
                    • Correct Answer: <code>{correct_ans}</code>
                    <div style="background: rgba(0, 0, 0, 0.2); padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.9rem;">
                        <strong>Step-by-step Solution:</strong><br>
                        {explanation_html}
                    </div>
                </div>
                """ , unsafe_allow_html=True)
                
            if st.button("🔄 Reset Quiz & Retake", use_container_width=True):
                st.session_state.aptitude_quiz = None
                st.session_state.aptitude_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()
