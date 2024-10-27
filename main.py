import os
import json
import sqlite3
import streamlit as st
# from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from openai.lib._parsing._completions import type_to_response_format_param
from langchain.output_parsers import PydanticOutputParser
import base64
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from docx import Document
from io import StringIO

class CategoryScore(BaseModel):
    score: float = Field(description="The score given for the category, out of 10")
    remarks: str = Field(description="Additional comments or feedback regarding the category")

class CVScores(BaseModel):
    education: CategoryScore
    skills: CategoryScore
    work_experience: CategoryScore
    overall_score: float = Field(description="The overall score calculated from the individual scores")

class CandidateFilter(BaseModel):
    universities: Optional[List[str]] = Field(description="List of universities")
    skills: Optional[List[str]] = Field(description="List of skills")
    experience_years: Optional[float] = Field(description="Minimum years of experience")
    names: Optional[List[str]] = Field(description="List of candidate names")
    field_of_study: Optional[str] = Field(description="Field of study")
    degree_level: Optional[str] = Field(description="Degree level, e.g., Bachelor's, Master's")


working_dir = os.path.dirname(os.path.abspath(__file__))

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    # embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

def display_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="750" height="1000" type="application/pdf"></iframe>'
    st.download_button(label="Download CV PDF", data=open(file_path, "rb"), file_name=os.path.basename(file_path), mime="application/pdf")
    st.write("Or view the CV below:")
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_docx(file_path):
    document = Document(file_path)
    doc_text = StringIO()
    for para in document.paragraphs:
        doc_text.write(para.text + '\n')
    st.text_area("DOCX Preview", value=doc_text.getvalue(), height=500)
    st.download_button(label="Download CV DOCX", data=open(file_path, "rb"), file_name=os.path.basename(file_path), mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

def adjust_sliders(changed_value, slider1, slider2):
    if changed_value == 100:
        return 0, 0
    remaining_value = 100 - changed_value
    if slider1 + slider2 == 0:
        return remaining_value // 2, remaining_value // 2
    ratio1 = slider1 / (slider1 + slider2)
    ratio2 = slider2 / (slider1 + slider2)
    return round(remaining_value * ratio1), round(remaining_value * ratio2)

def analyze_and_rank_cv(cv_text, job_description, analyze_chain):
    prompt_template = """
    Analyze the following CV and extract the key information relevant to the job description provided.

    CV Text:
    {cv_text}

    Job Description:
    {job_description}

    Provide scores for the following categories:
    - Education (out of 10)
    - Skills (out of 10)
    - Work Experience (out of 10)
    
    Also provide a recommendation for improvement.
    """
    
    prompt = PromptTemplate(
        input_variables=["cv_text", "job_description"],
        template=prompt_template
    )
    
    combined_input = prompt.format(cv_text=cv_text, job_description=job_description)
    
    response = analyze_chain.invoke(combined_input)
    
    return response.content

def extract_scores_from_detailed_text(detailed_text, extraction_chain):
    prompt_template = """
    Extract the following information from the detailed CV analysis:

    Detailed CV Analysis:
    {detailed_text}

    Please return the scores in a JSON format with the following structure:
    {{
        "education": {{
            "score": <integer from 1 to 10>,
            "remarks": "<short summary of the education>"
        }},
        "skills": {{
            "score": <integer from 1 to 10>,
            "remarks": "<short summary of the skills>"
        }},
        "work_experience": {{
            "score": <integer from 1 to 10>,
            "remarks": "<short summary of the work experience>"
        }},
        "overall_score": <overall score as an integer from 1 to 10>
    }}

    Make sure that the JSON is in the exact format specified above without additional nested objects or lists.
    """

    prompt = PromptTemplate(
        input_variables=["detailed_text"],
        template=prompt_template
    )

    llm_structural_chain = prompt | extraction_chain.with_structured_output(CVScores)

    response = llm_structural_chain.invoke({"detailed_text": detailed_text})
    
    return response

def query_cvs_from_db():
    conn = sqlite3.connect('cv_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, name, education, skills, work_experience FROM cvs")
    return cursor.fetchall()

def query_candidates_from_db(name=None, education=None, min_experience_years=None, skills=None, field_of_study=None, degree_level=None):
    conn = sqlite3.connect('cv_data.db')
    cursor = conn.cursor()

    query = "SELECT * FROM cvs WHERE 1=1"
    params = []

    # Name
    if name and isinstance(name, list) and len(name) > 0:
        name_conditions = " OR ".join(["name LIKE ?"] * len(name))
        query += f" AND ({name_conditions})"
        for partial_name in name:
            params.append(f"%{partial_name}%")
    # Education
    if education and isinstance(education, list) and len(education) > 0:
        education_conditions = " OR ".join(["education LIKE ?"] * len(education))
        query += f" AND ({education_conditions})"
        for university in education:
            params.append(f"%{university}%")
    # Skills
    if skills and isinstance(skills, list) and len(skills) > 0:
        for skill in skills:
            query += " AND skills LIKE ?"
            params.append(f"%{skill}%")
    # Experience years
    if min_experience_years is not None:
        query += " AND total_experience_years >= ?"
        params.append(min_experience_years)
    # Field of study
    if field_of_study and isinstance(field_of_study, str):
        query += " AND education LIKE ?"
        params.append(f"%{field_of_study}%") 
    # Degree level
    if degree_level and isinstance(degree_level, str):
        query += " AND education LIKE ?"
        params.append(f"%{degree_level}%")

    print(f"Query: {query}")
    print(f"Params: {params}")

    cursor.execute(query, params)
    candidates = cursor.fetchall()

    conn.close()

    return candidates

def interpret_query_with_llm(query, llm_structural_chain):
    prompt_template = """
    You are tasked with interpreting user queries for filtering CVs in a database. Based on the user's request, extract filtering criteria such as university, skills, experience years, or name, as well as education details like degree level and field of study.

    Here is the query to interpret:
    {query}

    Here are examples of possible queries:
    - "List candidates who graduated from top 10 universities in Turkey."
    - "I want candidates with Python and SQL skills who have at least 3 years of experience."
    - "Show me candidates who studied at Stanford University with a degree in Computer Science."
    - "Find candidates who graduated from Harvard and MIT, and have worked with AI and data science."

    You need to interpret the query and return the following structure:

    {{
        "universities": ["<list of universities or None>"],
        "skills": ["<list of skills or None>"],
        "experience_years": <minimum years of experience or 0>,
        "names": ["<list of candidate names or None>"],
        "field_of_study": "<field of study or None>",
        "degree_level": "<degree level or None>"
    }}

    Make sure to:
    - Extract only relevant information for filtering.
    - Return null values for fields not specified in the query.
    - If the user mentions a complex condition like "top 10 universities," convert it into a list of universities.
    - Return each field as a list, even if there's only one item.
    - For education, make sure to include the degree level, field of study, and university name if mentioned.

    For example, for the query "I want candidates who graduated from the top 5 universities in Turkey and know Python," return:
    {{
        "universities": ["Bogazici University", "Middle East Technical University", "Istanbul University", "Koç University", "Sabanci University"],
        "skills": ["Python"],
        "experience_years": None,
        "names": [],
        "field_of_study": None,
        "degree_level": None
    }}

    Another example:
    Query: "List candidates who have 5 years of experience, studied at Stanford University, and know React."
    Output:
    {{
        "universities": ["Stanford University"],
        "skills": ["React"],
        "experience_years": 5,
        "names": [],
        "field_of_study": None,
        "degree_level": None
    }}

    If the query is for specific candidates by name, return their names in the "names" field.

    If you find a single item, return it as a list with one item.

    Ensure the output is in valid JSON format and matches the structure provided.
    """
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template
    )

    output_parser = PydanticOutputParser(pydantic_object=CandidateFilter)

    llm_with_tools = llm_structural_chain.bind_tools([CandidateFilter], strict=True)
    
    llm_structural_chain = prompt | llm_with_tools | output_parser

    response = llm_structural_chain.invoke({"query": query})

    return response

# Setting up the chat chain
def chat_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    return chain

def dynamic_context_aware_chat(query, chat_chain):
    if isinstance(query, list):
        query = " ".join(query)
    elif not isinstance(query, str):
        query = str(query)
    
    response = chat_chain({"question": query})
    
    relevant_cv_chunks = response.get('source_documents', [])
    
    for chunk in relevant_cv_chunks:
        print(f"Relevant CV Chunk: {chunk.page_content}")
    
    return response["answer"]

def analyze_chain():
    llm = ChatOpenAI(model="gpt-4o",temperature=0)
    
    return llm

def extraction_chain_openai():
    llm = ChatOpenAI(model="gpt-4o",temperature=0)
        
    return llm

def extraction_chain_ollama():
    llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0, format="json")
        
    return llm

############################
######### Streamlit ########
############################
st.set_page_config(
    page_title="Multi Doc Chat and CV Ranking",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi Documents Chat and CV Ranking")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

if "extraction_chain" not in st.session_state:
    st.session_state.extraction_chain = extraction_chain_openai()
    
if "analyze_chain" not in st.session_state:
    st.session_state.analyze_chain = analyze_chain()

if 'generated' not in st.session_state:
    st.session_state.generated = []

if 'past' not in st.session_state:
    st.session_state.past = []
    
if 'education_weight' not in st.session_state:
    st.session_state.education_weight = 33

if 'skills_weight' not in st.session_state:
    st.session_state.skills_weight = 33

if 'work_experience_weight' not in st.session_state:
    st.session_state.work_experience_weight = 34

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 CV Ranking", "🔍 CV Filtering"])

import streamlit.components.v1 as components

# Tab 1: Multi Documents Chat
with tab1:
    if 'past' in st.session_state and 'generated' in st.session_state:
        chat_container = st.container()
        with chat_container:
            for i in range(len(st.session_state['past'])):
                with st.chat_message("user"):
                    st.markdown(st.session_state['past'][i])
                with st.chat_message("assistant"):
                    st.markdown(st.session_state['generated'][i])

    auto_scroll_js = """
        <script>
            var chatContainer = document.getElementsByClassName('stContainer')[0];
            chatContainer.scrollTop = chatContainer.scrollHeight;
        </script>
    """
    components.html(auto_scroll_js)

    st.write("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    user_input = st.chat_input("Ask AI...")

    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        st.session_state.past.append(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner('Generating response...'):
            assistant_response = dynamic_context_aware_chat(user_input, st.session_state.conversational_chain)
        
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

        st.session_state.generated.append(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    st.write("<div style='height: 50px;'></div>", unsafe_allow_html=True)

# Tab 2: CV Ranking
with tab2:
    st.header("📊 CV Ranking Based on Job Description")

    cvs = query_cvs_from_db()
    
    job_description = st.text_area("Enter the job description", height=150)

    education_weight = st.session_state.education_weight
    skills_weight = st.session_state.skills_weight
    work_experience_weight = st.session_state.work_experience_weight

    st.write("Adjust the sliders to set weights (they must sum to 100%):")

    education_weight = st.slider(
        "Education Weight (%)",
        min_value=0,
        max_value=100,
        value=education_weight,
        key="education_slider"
    )

    if education_weight != st.session_state.education_weight:
        skills_weight, work_experience_weight = adjust_sliders(education_weight, skills_weight, work_experience_weight)

    skills_weight = st.slider(
        "Skills Weight (%)",
        min_value=0,
        max_value=100,
        value=skills_weight,
        key="skills_slider"
    )

    if skills_weight != st.session_state.skills_weight:
        education_weight, work_experience_weight = adjust_sliders(skills_weight, education_weight, work_experience_weight)

    work_experience_weight = st.slider(
        "Work Experience Weight (%)",
        min_value=0,
        max_value=100,
        value=work_experience_weight,
        key="work_experience_slider"
    )

    if work_experience_weight != st.session_state.work_experience_weight:
        education_weight, skills_weight = adjust_sliders(work_experience_weight, education_weight, skills_weight)

    st.session_state.education_weight = education_weight
    st.session_state.skills_weight = skills_weight
    st.session_state.work_experience_weight = work_experience_weight

    st.write(f"Education Weight: {education_weight}%")
    st.write(f"Skills Weight: {skills_weight}%")
    st.write(f"Work Experience Weight: {work_experience_weight}%")

    total_weight = education_weight + skills_weight + work_experience_weight
    if total_weight != 100:
        st.warning(f"The weights must add up to 100%. Currently, they add up to {total_weight}%.")

    # Rank all CVs based on the job description
    if job_description and total_weight == 100:
        if st.button("Rank All CVs"):
            ranked_candidates = []
            for cv in cvs:
                cv_text = f"Name: {cv[1]}\nEducation: {cv[2]}\nSkills: {cv[3]}\nWork Experience: {cv[4]}"
                analyze_chain = st.session_state.analyze_chain
                extraction_chain = st.session_state.extraction_chain
                
                detailed_response = analyze_and_rank_cv(cv_text, job_description, analyze_chain)
                
                scores = extract_scores_from_detailed_text(detailed_response, extraction_chain)
                
                education_score = scores.education.score
                skills_score = scores.skills.score
                work_experience_score = scores.work_experience.score
                
                overall_score = (
                    (education_score * education_weight) +
                    (skills_score * skills_weight) +
                    (work_experience_score * work_experience_weight)
                ) / 100
                
                ranked_candidates.append({
                    "name": cv[1],
                    "education_score": education_score,
                    "skills_score": skills_score,
                    "work_experience_score": work_experience_score,
                    "overall_score": overall_score,
                    "file_name": cv[0],
                    "details": detailed_response
                })

            ranked_candidates = sorted(
                ranked_candidates, 
                key=lambda x: float(x["overall_score"]),
                reverse=True
            )

            st.write("### Ranked Candidates")
            for i, candidate in enumerate(ranked_candidates, start=1):
                st.write(f"**Rank {i}: {candidate['name']}** - Overall Score: {candidate['overall_score']}")
                st.write(f"Education Score: {candidate['education_score']}, Skills Score: {candidate['skills_score']}, Work Experience Score: {candidate['work_experience_score']}")
                
                with st.expander(f"View {candidate['name']}'s CV"):
                    if candidate['file_name'].endswith('.pdf'):
                        display_pdf(f"data/{candidate['file_name']}")
                    elif candidate['file_name'].endswith('.docx'):
                        display_docx(f"data/{candidate['file_name']}")

                with st.expander(f"View detailed analysis for {candidate['name']}"):
                    st.write(candidate["details"])
                    
# Tab 3: CV Filtering
with tab3:
    st.header("🔍 CV Filtering")

    col1, col2 = st.columns([1, 3])
    
    extraction_chain = st.session_state.extraction_chain

    with col1:
        st.subheader("AI Filter")
        ai_filter_query = st.text_area("Enter a query")

        if st.button("Apply Filters"):
            if ai_filter_query:
                interpreted_query = interpret_query_with_llm(ai_filter_query, extraction_chain)
                
                print("*****************",interpreted_query, "*****************")
                
                education = getattr(interpreted_query, "universities", None)
                experience_years = getattr(interpreted_query, "experience_years", None)
                skills = getattr(interpreted_query, "skills", None)
                names = getattr(interpreted_query, "names", None)
                field_of_study = getattr(interpreted_query, "field_of_study", None)
                degree_level = getattr(interpreted_query, "degree_level", None)
                print(f"Universities: {education}")
                print(f"Skills: {skills}")
                print(f"Experience Years: {experience_years}")
                print(f"Names: {names}")
                filtered_candidates = query_candidates_from_db(name=names, 
                                                               education=education, 
                                                               min_experience_years=experience_years,
                                                               skills=skills,
                                                               field_of_study=field_of_study,
                                                               degree_level=degree_level)

                with col2:
                    st.subheader(f"Found {len(filtered_candidates)} candidates")
                    if filtered_candidates:
                        for candidate in filtered_candidates:
                            st.write(f"Candidate: {candidate[2]}")
                            with st.expander(f"View {candidate[2]}'s CV"):
                                if candidate[1].endswith('.pdf'):
                                    display_pdf(f"data/{candidate[1]}")
                                elif candidate[1].endswith('.docx'):
                                    display_docx(f"data/{candidate[1]}")
                    else:
                        st.write("No candidates found matching the filter.")