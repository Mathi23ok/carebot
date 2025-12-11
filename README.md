🌿 CareSpace — RAG-Powered Emotional Support Chatbot

CareSpace is a lightweight mental-health support chatbot built using FastAPI, Streamlit, SentenceTransformers, FAISS, and a local Ollama LLM.
It combines retrieval-augmented generation (RAG) with an empathy-focused prompt to deliver short, supportive responses.

✨ Features

RAG Pipeline: Retrieves the most relevant supportive snippet using MiniLM embeddings + FAISS.
Local LLM (Ollama): Generates calm and empathetic responses using a fast on-device model like phi3.
Safety Layer: Detects crisis phrases and returns an immediate non-LLM emergency message.
Simple UI: Clean Streamlit chat interface.
Fallback Handling: If Ollama fails, the system returns a safe template response.

🚀 How to Run
1. Install dependencies
pip install fastapi uvicorn streamlit sentence-transformers faiss-cpu requests

2. Install and pull an Ollama model
ollama pull phi3

3. Start the backend
python backend/main.py

4. Start the Streamlit UI
streamlit run streamlit_client.py

🧠 How It Works

User sends a message from the Streamlit UI.
Backend embeds the text and retrieves a relevant supportive line from a small knowledge base.
A prompt is built combining user message + retrieved context.
Ollama generates an empathetic reply.
Streamlit displays the conversation.

🛡️ Crisis Detection

If the user types phrases like:
“I want to die”
“kill myself”
“I can’t go on”
The bot does not generate with the LLM.
It immediately returns a safe crisis-escalation message.

📁 Project Structure
carebot/
 ├─ backend/main.py          # FastAPI backend + RAG + Ollama calling
 ├─ streamlit_client.py      # Streamlit chat UI
 ├─ README.md
 └─ requirements.txt

📌 Tech Stack

Python
FastAPI
Streamlit
SentenceTransformers (MiniLM)
FAISS
Ollama
Uvicorn

🌼 Notes

This project is meant for learning AI, RAG, embeddings, and chatbot UX.
It is not a medical tool.
