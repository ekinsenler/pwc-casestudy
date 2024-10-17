import os
import sqlite3
import json
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama


# Initialize LLM (Ollama)
llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0, format="json")

# Initialize database
def init_db():
    conn = sqlite3.connect('cv_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS cvs 
                      (id INTEGER PRIMARY KEY, file_name TEXT, name TEXT, education TEXT, skills TEXT, work_experience TEXT)''')
    conn.commit()
    return conn


def extract_cv_info_using_llm(cv_text, llm):
    schema_prompt_template = """
    You are an expert at extracting structured information from documents. 

    Please extract the following fields from the CV:
    - Name
    - Education (degree, university, year)
    - Skills
    - Work experience (company, role, years of experience)

    CV Text:
    {cv_text}

    Return the information in the following JSON format:
    {{
      "name": "",
      "education": {{
        "degree": "",
        "university": "",
        "year": ""
      }},
      "skills": [],
      "work_experience": [
        {{
          "company": "",
          "role": "",
          "years_of_experience": ""
        }}
      ]
    }}
    """
    
    # Define the prompt template
    prompt = PromptTemplate(input_variables=["cv_text"], template=schema_prompt_template)
    
    # Create a runnable sequence that pipes the prompt into the LLM
    chain = prompt | llm
    
    # Invoke the chain with the CV text
    response = chain.invoke({"cv_text": cv_text})
    
    # Parse the response as JSON and return it
    return json.loads(response.content)


# Function to save extracted data to the database
def save_extracted_data_to_db(conn, structured_cv_info, file_name):
    cursor = conn.cursor()
    name = structured_cv_info["name"]
    education = json.dumps(structured_cv_info["education"])
    skills = json.dumps(structured_cv_info["skills"])
    work_experience = json.dumps(structured_cv_info["work_experience"])
    cursor.execute("INSERT INTO cvs (file_name, name, education, skills, work_experience) VALUES (?, ?, ?, ?, ?)",
                   (file_name, name, education, skills, work_experience))
    conn.commit()

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Load the documents from the directory
loader = DirectoryLoader(path="data", glob="./*.pdf", loader_cls=UnstructuredFileLoader)
documents = loader.load()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

# Initialize the database connection
conn = init_db()

# Iterate over each document, extract structured information, and save to database
for document in documents:
    file_name = os.path.basename(document.metadata["source"])  # Assuming document has metadata containing file path
    cv_text = document.page_content  # Extract full text of the document

    # Extract structured information from the CV using LLM
    structured_cv_info = extract_cv_info_using_llm(cv_text, llm)

    # Save the structured information to the database
    save_extracted_data_to_db(conn, structured_cv_info, file_name)

# Vectorize the text chunks and persist them in Chroma
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

print("Documents Vectorized and Key Information Saved")
