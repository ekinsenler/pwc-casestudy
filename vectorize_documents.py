import os
import sqlite3
import json
import datetime
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
# from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import getpass
load_dotenv()


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Initialize LLM (Ollama)
llm = ChatOpenAI(model="gpt-4o",temperature=0, model_kwargs={"response_format":{"type": "json_object"}})

# Initialize database
def init_db():
    conn = sqlite3.connect('cv_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS cvs 
                      (id INTEGER PRIMARY KEY, file_name TEXT, name TEXT, education TEXT, skills TEXT, work_experience TEXT, total_experience_years REAL)''')
    conn.commit()
    return conn
  
  # Helper function to calculate total years of experience
def calculate_total_experience(work_experience):
    total_experience_years = 0.0
    for experience in work_experience:
        try:
            total_experience_years += float(experience["years_of_experience"])
        except ValueError:
            continue  # Skip any invalid date format entries
    return total_experience_years


def extract_cv_info_using_llm(cv_text, llm):
    schema_prompt_template = """
    You are an expert at extracting structured information from documents and translating fields into English. Additionally, convert any Turkish characters (such as ö, ç, ı, ğ, ü, ş) into their English equivalents for consistency.

    Please extract the following fields from the CV and translate them into English:
    - Name
    - Education (university name, field of study, degree level, year)
    - Skills
    - Work experience (company, role, years of experience)

    If any of the fields are in Turkish or a language other than English, translate them into English and convert any Turkish characters into English characters. 

    CV Text:
    {cv_text}

    If the university name is present but the degree is not mentioned, please assume that the degree is a Bachelor's degree.

    Return the information in the following JSON format:
    {{
      "name": "",
      "education": [
        {{
          "university_name": "",
          "field_of_study": "",
          "degree_level": "",
          "year": ""
        }}
      ],
      "skills": [],
      "work_experience": [
        {{
          "company": "",
          "role": "",
          "years_of_experience": ""
        }}
      ]
    }}

    Important:
    - Translate any non-English field into English.
    - Convert Turkish characters (e.g., ö, ç, ı, ğ, ü, ş) into their English equivalents (e.g., o, c, i, g, u, s).
    - Ensure that the JSON output is structured with the fields translated and standardized using English characters.

    Example:

    CV Text:
    John Doe is a software engineer with expertise in Python, Java, and machine learning. 
    He graduated from "İstanbul Üniversitesi" with a "Lisans Bilgisayar Mühendisliği" (Bachelor's in Computer Engineering) in 2017. 
    John worked at "Söğüş Yazılım" as a "Kıdemli Yazılım Mühendisi" for 5 years and later joined "Facebook" as an "Engineering Manager," where he spent 3 years.
    
    Expected JSON output:
    {{
      "name": "John Doe",
      "education": [
        {{
          "university_name": "Istanbul University",
          "field_of_study": "Computer Engineering",
          "degree_level": "Bachelor's",
          "year": "2017"
        }}
      ],
      "skills": ["Python", "Java", "Machine Learning"],
      "work_experience": [
        {{
          "company": "Sogus Yazilim",
          "role": "Senior Software Engineer",
          "years_of_experience": "5"
        }},
        {{
          "company": "Facebook",
          "role": "Engineering Manager",
          "years_of_experience": "3"
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
    
    # Extract data from the structured_cv_info
    name = structured_cv_info["name"]
    education = json.dumps(structured_cv_info["education"])
    skills = json.dumps(structured_cv_info["skills"])
    work_experience = json.dumps(structured_cv_info["work_experience"])
    
    # Calculate total experience years
    total_experience_years = calculate_total_experience(structured_cv_info["work_experience"])
    
    # Insert data into the database
    cursor.execute("""
        INSERT INTO cvs (file_name, name, education, skills, work_experience, total_experience_years)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (file_name, name, education, skills, work_experience, total_experience_years))
    
    conn.commit()

# Function to serialize metadata to JSON string
def serialize_metadata(metadata):
    return json.dumps(metadata)

# Initialize the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load the documents from the directory
loader = DirectoryLoader(path="data", glob=["./*.pdf", "./*.doc", "./*.docx"], loader_cls=UnstructuredFileLoader)
documents = loader.load()

# Initialize the database connection
conn = init_db()

# Iterate over each document, extract structured information, and save to database
all_chunks_with_metadata = []
for document in documents:
    file_name = os.path.basename(document.metadata["source"])  # Assuming document has metadata containing file path
    cv_text = document.page_content  # Extract full text of the document

    # Extract structured information from the CV using LLM
    structured_cv_info = extract_cv_info_using_llm(cv_text, llm)

    # Save the structured information to the database
    save_extracted_data_to_db(conn, structured_cv_info, file_name)

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    text_chunks = text_splitter.split_documents([document])

    # Serialize metadata to JSON string
    serialized_metadata = serialize_metadata(structured_cv_info)
    for chunk in text_chunks:
        chunk.metadata["structured_info"] = serialized_metadata
        all_chunks_with_metadata.append(chunk)

# Vectorize the text chunks with metadata and persist them in Chroma
vectordb = Chroma.from_documents(
    documents=all_chunks_with_metadata,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

print("Documents Vectorized and Key Information Saved")
