# backend/main.py
# CareBot — RAG + OpenAI backend

import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI

# ---------- setup ----------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="CareBot")

# ---------- embedding model ----------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------- small curated KB ----------
KB = [
    "When you feel overwhelmed, try 4-4-4 breathing: inhale 4s, hold 4s, exhale 4s.",
    "Grounding technique: name 5 things you can see, 4 you can touch, 3 you can hear.",
    "Labeling emotions helps: say 'I feel sad' or 'I feel angry' — naming it reduces intensity.",
    "If you're in immediate danger or thinking of harming yourself, contact emergency services right away."
]

kb_embeddings = embed_model.encode(KB, convert_to_numpy=True)
dim = kb_embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(kb_embeddings)

# ---------- request model ----------
class Message(BaseModel):
    text: str

# ---------- retrieval ----------
def retrieve(user_text: str, k: int = 1):
    vec = embed_model.encode([user_text], convert_to_numpy=True)
    _, idx = index.search(vec, k)
    return [KB[i] for i in idx[0]]

# ---------- prompt ----------
def build_prompt(user_text: str, retrieved: str):
    return f"""
You are a calm, empathetic mental health support assistant.
Validate emotions. Be kind. Offer one gentle coping suggestion.
Do NOT give medical advice.
Keep replies short (1–3 sentences).

Context:
{retrieved}

User:
{user_text}

Assistant:
""".strip()

# ---------- OpenAI call ----------
def query_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # fast + cheap + strong
        messages=[
            {"role": "system", "content": "You are a supportive mental health companion."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# ---------- API ----------
@app.post("/chat")
def chat(msg: Message):
    user = (msg.text or "").strip()

    if not user:
        return {"reply": "Hey — I didn't catch that. Want to try again?"}

    # --- crisis detection ---
    crisis_keywords = [
        "kill myself", "suicide", "want to die",
        "end my life", "hurt myself", "i can't go on"
    ]

    if any(k in user.lower() for k in crisis_keywords):
        return {
            "reply": (
                "I'm really sorry you're feeling this way. "
                "If you're in immediate danger, please contact your local emergency number or a suicide prevention hotline. "
                "You matter, and help is available."
            ),
            "escalation": True
        }

    # --- RAG ---
    retrieved = retrieve(user, k=1)[0]
    prompt = build_prompt(user, retrieved)

    try:
        reply = query_openai(prompt)
    except Exception as e:
        reply = f"I hear you. {retrieved} I'm here with you."

    return {
        "reply": reply,
        "retrieved": retrieved
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
