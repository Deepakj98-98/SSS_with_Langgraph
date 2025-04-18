# SSS: Smart Skill Support

SSS is an intelligent, role-aware chatbot system that processes multi-modal inputs (text, audio, images, documents), rephrases content based on user roles, and responds intelligently using LangGraph, Ollama, Whisper, OCR, and Qdrant. 
This application can also support in automating the upskilling or skill training for onboarded resources by curating content based on the user’s skill/role.

---

## 🚀 Features

- **Role-Specific Rephrasing** (Dev, Analyst, QA, Management)
- **Follow-Up Question Detection**
- **Multi-modal Uploads**: Supports text, images (OCR), audio (speech-to-text), PDFs, Word docs, etc.
- **Async-Powered LangGraph Pipelines**
- **LLM-Powered Reasoning** via [Ollama](https://ollama.com/)
- **Vector-Based Search** with Qdrant
- **Embeddings** using HuggingFace models

---

## 🧠 Tech Stack

| Component            | Tech Used                                |
|---------------------|-------------------------------------------|
| LLM Inference       | Ollama + Mistral / Gemma                  |
| Workflow Engine     | LangGraph (async, conditional branching)  |
| Storage             | MongoDB + GridFS                          |
| Vector DB           | Qdrant                                    |
| Embeddings          | `sentence-transformers`                   |
| Document Handling   | PyTesseract, pdf2image, python-docx       |
| Audio Transcription | Whisper + PyDub                           |
| Web Framework       | Flask (API + UI)                          |
| Async HTTP          | aiohttp                                   |

---

## 📦 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Deepakj98-98/SSS_with_Langgraph.git
