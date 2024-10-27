# Multi-Document Chat and CV Ranking System

This project is a multi-document chat and CV ranking system that allows users to interact with CV data for candidate filtering and ranking purposes. Powered by large language models (LLMs) for query interpretation and similarity matching, this system efficiently retrieves and ranks CVs based on user-defined criteria.

---

## Features

- **Interactive Chat Interface**: Engage with an LLM-powered chatbot for CV-related inquiries.
- **Advanced Filtering**: Filter candidates based on multiple attributes, including university, skills, degree level, field of study, and experience.
- **CV Ranking System**: Rank candidates based on a job description using adjustable category weights for education, skills, and work experience.
- **Dynamic PDF and DOCX Viewing**: Inline document viewers for easy access to candidate CVs.

## Requirements

- Python 3.8+
- Libraries: `streamlit`, `langchain`, `sqlite3`, `openai`, `docx`, `pydantic`

Install dependencies with:
```bash
pip install -r requirements.txt
```
## Setup and Run

1. **Database setup**: To vectorize document and setup the database that is necessary for the application:
    ```bash
    python vectorize_documents.py
    ```
2. **Run the application**: To launch the streamlit app:
    ```bash
    streamlit run main.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.