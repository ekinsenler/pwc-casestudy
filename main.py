import os
import json
import sqlite3
import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from io import BytesIO
import base64

# Load embedding model
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))


# Setup vectorstore
def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore


# Function to display PDF using base64 encoding
def display_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        pdf_bytes = base64.b64encode(pdf_file.read())
        pdf_stream = BytesIO(pdf_bytes)
        pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_stream}" width="700" height="1000" type="application/pdf"></iframe>'
        st.download_button(
            label="Download CV PDF",
            data=pdf_stream,
            file_name=os.path.basename(file_path),
            mime="application/pdf"
        )
        st.write("Or view the CV below:")
        st.markdown(pdf_display, unsafe_allow_html=True)


# Function to generate the detailed analysis of the CV
def analyze_and_rank_cv(cv_text, job_description, chain: ConversationalRetrievalChain):
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
    
    # Invoke the chain using "question" as the key
    response = chain.invoke({"question": combined_input})
    
    return response["answer"]  # Return the detailed response text


# Function to extract scores using another LLM from the detailed text
def extract_scores_from_detailed_text(detailed_text, extraction_chain):
    prompt_template = """
    Extract the following information from the detailed CV analysis:

    Detailed CV Analysis:
    {detailed_text}

    Provide scores for:
    - Education (out of 10)
    - Skills (out of 10)
    - Work Experience (out of 10)
    
    Return the extracted scores in JSON format with keys: education, skills, work_experience, overall_score.
    """
    
    prompt = PromptTemplate(
        input_variables=["detailed_text"],
        template=prompt_template
    )
    
    # Combine detailed_text into a single "question" field
    combined_input = prompt.format(detailed_text=detailed_text)
    
    # Invoke the extraction LLM using the combined input (as a string)
    response = extraction_chain.invoke(combined_input)
    
    # Parse and return the response as JSON
    return json.loads(response.content)  # Assuming the LLM returns a JSON string with the required fields




# Function to query CVs from the database
def query_cvs_from_db():
    conn = sqlite3.connect('cv_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, name, education, skills, work_experience FROM cvs")
    return cursor.fetchall()


# Setting up the chat chain
def chat_chain(vectorstore):
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
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


# Streamlit UI setup
st.set_page_config(
    page_title="Multi Doc Chat and CV Ranking",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi Documents Chat and CV Ranking")

# Initialize session states for chat history and vectorstore
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

# Corrected variable name for conversational_chain
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

if "extraction_chain" not in st.session_state:
    st.session_state.extraction_chain = ChatOllama(model="llama3.2:1b", temperature=0, format="json")

if 'generated' not in st.session_state:
    st.session_state.generated = []

if 'past' not in st.session_state:
    st.session_state.past = []


# Create tabs for the two functionalities
tab1, tab2 = st.tabs(["💬 Multi Documents Chat", "📊 CV Ranking"])

# Tab 1: Multi Documents Chat
with tab1:
    # Handle user input
    user_input = st.chat_input("Ask AI...")

    if user_input:
        # Immediately display user's input in the user container
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add user message to session state history
        st.session_state.past.append(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Generate the assistant's response and display in the assistant container
        with st.spinner('Generating response...'):
            output = st.session_state.conversational_chain.invoke({"question": user_input})
            assistant_response = output["answer"]
        
        # Display assistant's response in its own container
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Add assistant response to session state history
        st.session_state.generated.append(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

# Tab 2: CV Ranking
with tab2:
    st.header("📊 CV Ranking Based on Job Description")

    # Load CV data from the database
    cvs = query_cvs_from_db()
    
    # Input job description
    job_description = st.text_area("Enter the job description", height=150)

    # Rank all CVs based on the job description
    if job_description:
        if st.button("Rank All CVs"):
            ranked_candidates = []
            for cv in cvs:
                cv_text = f"Name: {cv[1]}\nEducation: {cv[2]}\nSkills: {cv[3]}\nWork Experience: {cv[4]}"
                chain = st.session_state.conversational_chain  # First LLM chain for analysis
                extraction_chain = st.session_state.extraction_chain  # Second LLM chain for extraction
                
                # Step 1: Generate detailed analysis using the first LLM
                detailed_response = analyze_and_rank_cv(cv_text, job_description, chain)
                
                # Step 2: Extract scores using the second LLM
                scores = extract_scores_from_detailed_text(detailed_response, extraction_chain)
                
                # Handle case where scores might be missing or invalid
                # overall_score = scores.get("overall_score", 0)
                education_score = scores.get("education", 0).get("score", 0)
                skills_score = scores.get("skills", 0).get("score", 0)
                work_experience_score = scores.get("work_experience", 0).get("score", 0)
                
                print(f"Scores for {cv[1]}: Education={education_score}, Skills={skills_score}, Work Experience={work_experience_score}")
                ranked_candidates.append({
                    "name": cv[1],
                    "education_score": education_score,
                    "skills_score": skills_score,
                    "work_experience_score": work_experience_score,
                    "overall_score": (education_score + skills_score + work_experience_score)/3,
                    "file_name": cv[0],
                    "details": detailed_response  # Store the detailed analysis for display if needed
                })

            
            # Ensure all scores are numeric before sorting
            ranked_candidates = sorted(
                ranked_candidates, 
                key=lambda x: float(x["overall_score"]),  # Ensure the score is a float
                reverse=True
            )

            # Display ranked candidates in a table with a link to view/download the CV
            st.write("### Ranked Candidates")
            for i, candidate in enumerate(ranked_candidates, start=1):
                st.write(f"**Rank {i}: {candidate['name']}** - Overall Score: {candidate['overall_score']}")
                st.write(f"Education Score: {candidate['education_score']}, Skills Score: {candidate['skills_score']}, Work Experience Score: {candidate['work_experience_score']}")
                
                # Create a button to view/download the CV
                with st.expander(f"View {candidate['name']}'s CV"):
                    display_pdf(f"data/{candidate['file_name']}")  # Assuming CVs are stored in a "data" folder

                # Optionally display detailed analysis
                with st.expander(f"View detailed analysis for {candidate['name']}"):
                    st.write(candidate["details"])
