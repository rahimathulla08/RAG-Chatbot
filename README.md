# College RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions from college-related PDF documents such as handbook, syllabus, internship guidelines, placement policy, and exam rules.

## Features
- Load multiple PDF documents
- Split documents into chunks
- Generate embeddings using Hugging Face
- Store vectors in FAISS
- Retrieve relevant chunks for user questions
- Generate answers using Groq LLM
- Streamlit-based chatbot UI
- Display source documents used for answers

## Tech Stack
- Python
- LangChain
- FAISS
- Hugging Face Embeddings
- Groq LLM
- Streamlit

## Project Structure
```bash
RAG-College_assistant/
│
├── data/
│   ├── aiml_syllabus.pdf
│   ├── college_handbook.pdf
│   ├── exam_rules.pdf
│   ├── internship_guidelines.pdf
│   └── placement_policy.pdf
│
├── vectorstore/
│   └── faiss_index/
│
├── ingest.py
├── query.py
├── app.py
├── requirements.txt
├── .env
└── README.md