import json
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of models to try in order (primary followed by fallbacks)
HF_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-3B-Instruct"
]

def _get_hf_response(system_instruction: str, prompt: str, api_key: str) -> str:
    """
    Helper function to query Hugging Face Inference API with automatic fallback.
    Uses direct HTTP requests to bypass strict router requirements.
    """
    if not api_key:
        return "Error: Hugging Face API token is missing."
        
    errors = []

    for model_name in HF_MODELS:
        try:
            logger.info(f"Attempting inference with model: {model_name}")
            api_url = f"https://api-inference.huggingface.co/models/{model_name}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_msg = response.text
                logger.warning(f"Model {model_name} failed: HTTP {response.status_code} - {error_msg}")
                errors.append(f"{model_name}: HTTP {response.status_code}")
                continue
                
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            errors.append(f"{model_name}: {str(e)}")
            continue
            
    return "Error: All Hugging Face fallback models failed. Details: " + " | ".join(errors)

def generate_rag_answer(query, retrieved_docs, api_key, **kwargs):
    """
    Generates a RAG response from Hugging Face Inference API.
    """
    context_str = ""
    for idx, doc in enumerate(retrieved_docs):
        src = doc.metadata.get("source", "Unknown Source")
        pg = doc.metadata.get("page", "?")
        context_str += f"\n--- Source Document {idx+1} [File: {src}, Page: {pg}] ---\n{doc.page_content}\n"

    system_instruction = (
        "You are an expert Placement Preparation Assistant and Technical Interviewer.\n"
        "Your goal is to help students prepare for campus placements, technical interviews, and core CS subjects.\n"
        "Answer the user's question using ONLY the provided text segments from retrieved documents.\n"
        "If the retrieved context does not contain enough information to answer, state that clearly, "
        "and then provide a comprehensive response using your general knowledge, clearly labeling the general knowledge part.\n"
        "Keep your answers structured, professional, and easy to read. Use code snippets in Java/Python where applicable.\n"
        "Use formatting like bullet points, bold text, and numbered lists."
    )

    prompt = f"User Query: {query}\n\nRetrieved Context Documents:\n{context_str}\n\nAnswer:"
    
    return _get_hf_response(system_instruction, prompt, api_key)


def analyze_resume(resume_text, api_key, **kwargs):
    """
    Analyzes the resume text using Hugging Face Inference API.
    """
    system_instruction = (
        "You are an expert HR Manager and Technical Recruiter.\n"
        "Analyze the student's resume. Identify their core skills, projects, and educational background.\n"
        "Provide a comprehensive, professional evaluation of the profile."
    )

    prompt = f"""
    Please analyze the following resume text:
    ---
    {resume_text}
    ---
    
    Provide the analysis structured in Markdown with these exact sections:
    
    ### 1. Profile Strength & Matching Score
    Provide a score out of 100 (e.g. 75/100) and list major highlights/relevance.
    
    ### 2. Core Skills Identified
    Group them into Technical (Programming languages, databases, developer tools) and Soft Skills/Methodologies.
    
    ### 3. Gap Analysis & Suggested Skills
    Identify missing skills that are standard for modern software engineering (SDE), web development, or analyst placement roles, and suggest which ones they should prioritize.
    
    ### 4. Placement Preparation Path
    Suggest actionable next steps (projects to build, subjects to study like OS/DBMS, and interview practices).
    
    ### 5. Personalized Interview Questions (with Model Answers)
    Generate 5 customized interview questions based on the resume's projects or skills:
    - 3 Technical Questions (related to projects/languages mentioned)
    - 2 Behavioral/HR Questions (related to teamwork, challenges, leadership)
    For each question, provide a detailed but concise model answer that the student can study.
    """

    return _get_hf_response(system_instruction, prompt, api_key)


def generate_company_prep(company_name, company_docs_text, api_key, **kwargs):
    """
    Generates company preparation checklist.
    """
    system_instruction = (
        "You are a Senior Placement Mentor. Your job is to analyze company preparation materials "
        "and suggest how a candidate can prepare specifically to clear that company's hiring process."
    )

    docs_context = f"\nCompany Specific Resources:\n{company_docs_text}\n" if company_docs_text else "\nNo specific files uploaded. Using general historical data for this company.\n"

    prompt = f"""
    Target Company: {company_name}
    {docs_context}
    
    Provide a detailed company preparation guide structured in Markdown with these exact sections:
    
    ### 1. Selection Process Overview
    Describe the typical recruitment stages (e.g., Aptitude Assessment, Online Coding Test, Technical Interview Rounds, Managerial/HR rounds) for {company_name}.
    
    ### 2. Key Subjects & Topics to Master
    List specific technical concepts (e.g. OOPS, specific algorithms, SQL joins, system design, system calls, networking protocols) that {company_name} frequently tests.
    
    ### 3. Top 5 Frequently Asked Technical/Coding Questions
    Provide 5 typical interview questions (with short code snippets or solutions) asked during {company_name} interviews.
    
    ### 4. Assessment Strategy & Pro-Tips
    Actionable tips to clear the online rounds and make a lasting impression in face-to-face interviews.
    """

    return _get_hf_response(system_instruction, prompt, api_key)


def generate_aptitude_quiz(topic, context_docs, num_questions, api_key, **kwargs):
    """
    Generates a structured MCQ aptitude quiz.
    """
    system_instruction = (
        "You are an expert Quantitative Aptitude and Logical Reasoning Instructor.\n"
        "Your task is to generate a multiple-choice quiz based on the retrieved document context.\n"
        "You MUST output raw JSON matching the specified structure, without any additional text. "
        "Format the output strictly as a JSON array of objects."
    )

    context_str = ""
    for idx, doc in enumerate(context_docs):
        context_str += f"\n--- Context Fragment {idx+1} ---\n{doc.page_content}\n"

    prompt = f"""
    Generate an aptitude quiz with {num_questions} questions on the topic: '{topic}'.
    Make sure the questions are representative of placement examinations.
    
    Use the following document text as inspiration if relevant:
    {context_str}
    
    You must output a JSON array where each item represents a question. 
    Each object MUST have the following keys:
    1. "id" (integer)
    2. "question" (string)
    3. "options" (array of 4 strings)
    4. "answer" (string, must exactly match one of the 4 options)
    5. "explanation" (string, step-by-step mathematical reasoning or logic explanation)
    
    Output ONLY valid JSON. Start with [ and end with ]. Do not wrap it in any HTML tags or markdown formatting other than optionally ```json.
    """

    def _parse_json(text: str):
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
        return json.loads(text)

    response_text = _get_hf_response(system_instruction, prompt, api_key)
    
    try:
        if not response_text.startswith("Error"):
            return _parse_json(response_text)
    except Exception as e:
        logger.error(f"Failed to parse JSON response: {e}")
        
    # Fallback data if API or parsing fails
    return [
        {
            "id": 1,
            "question": "A train 100 meters long is running at the speed of 30 km/hr. Find the time taken by it to pass a man standing near the railway line.",
            "options": ["12 seconds", "15 seconds", "10 seconds", "18 seconds"],
            "answer": "12 seconds",
            "explanation": "Speed = 30 km/hr = 30 * (5/18) m/s = 25/3 m/s.\nDistance = 100 meters.\nTime taken = Distance / Speed = 100 / (25/3) = 100 * 3 / 25 = 12 seconds."
        },
        {
            "id": 2,
            "question": "A worker is paid $150 for 5 days of work. How much will he be paid if he works for 20 days at the same rate?",
            "options": ["$400", "$600", "$500", "$800"],
            "answer": "$600",
            "explanation": "Daily wage = $150 / 5 = $30 per day.\nWage for 20 days = $30 * 20 = $600."
        }
    ]
