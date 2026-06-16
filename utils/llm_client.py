import json
import google.generativeai as genai
from openai import OpenAI

def generate_rag_answer(query, retrieved_docs, api_key, provider="Google Gemini", model_name="gemini-1.5-flash"):
    """
    Generates a RAG response from Google Gemini, xAI Grok, or Local Ollama.
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
    
    if provider == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {str(e)}"
    elif provider == "xAI Grok":
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error communicating with Grok API: {str(e)}"
    elif provider == "Local Ollama":
        try:
            import ollama
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with local Ollama: {str(e)}. Make sure Ollama server is running (run 'ollama run {model_name}')."

def analyze_resume(resume_text, api_key, provider="Google Gemini", model_name="gemini-1.5-flash"):
    """
    Analyzes the resume text using Gemini, Grok, or local Ollama.
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
    
    if provider == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {str(e)}"
    elif provider == "xAI Grok":
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error communicating with Grok API: {str(e)}"
    elif provider == "Local Ollama":
        try:
            import ollama
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with local Ollama: {str(e)}. Make sure Ollama server is running (run 'ollama run {model_name}')."

def generate_company_prep(company_name, company_docs_text, api_key, provider="Google Gemini", model_name="gemini-1.5-flash"):
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
    
    if provider == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {str(e)}"
    elif provider == "xAI Grok":
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error communicating with Grok API: {str(e)}"
    elif provider == "Local Ollama":
        try:
            import ollama
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with local Ollama: {str(e)}. Make sure Ollama server is running (run 'ollama run {model_name}')."

def generate_aptitude_quiz(topic, context_docs, num_questions, api_key, provider="Google Gemini", model_name="gemini-1.5-flash"):
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
    
    if provider == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            
            # Parse the JSON response
            text = response.text.strip()
            
            # Remove markdown wrappers
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            start_idx = text.find("[")
            end_idx = text.rfind("]")
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx+1]
                
            quiz_data = json.loads(text)
            return quiz_data
        except Exception as e:
            print(f"Error parsing Gemini aptitude quiz JSON: {e}")
            
    elif provider == "xAI Grok":
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            
            text = completion.choices[0].message.content.strip()
            
            # Remove markdown wrappers
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            start_idx = text.find("[")
            end_idx = text.rfind("]")
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx+1]
                
            quiz_data = json.loads(text)
            return quiz_data
        except Exception as e:
            print(f"Error parsing Grok aptitude quiz JSON: {e}")
            
    elif provider == "Local Ollama":
        try:
            import ollama
            response = ollama.chat(
                model=model_name,
                format="json", # Instruct Ollama to output valid JSON
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response['message']['content'].strip()
            quiz_data = json.loads(text)
            return quiz_data
        except Exception as e:
            print(f"Error generating quiz via local Ollama: {e}")

    # Fallback data if API or parsing fails
    return [
        {
            "id": 1,
            "question": f"A train 100 meters long is running at the speed of 30 km/hr. Find the time taken by it to pass a man standing near the railway line.",
            "options": [
                "12 seconds",
                "15 seconds",
                "10 seconds",
                "18 seconds"
            ],
            "answer": "12 seconds",
            "explanation": "Speed = 30 km/hr = 30 * (5/18) m/s = 25/3 m/s.\nDistance = 100 meters.\nTime taken = Distance / Speed = 100 / (25/3) = 100 * 3 / 25 = 12 seconds."
        },
        {
            "id": 2,
            "question": "A worker is paid $150 for 5 days of work. How much will he be paid if he works for 20 days at the same rate?",
            "options": [
                "$400",
                "$600",
                "$500",
                "$800"
            ],
            "answer": "$600",
            "explanation": "Daily wage = $150 / 5 = $30 per day.\nWage for 20 days = $30 * 20 = $600."
        }
    ]
