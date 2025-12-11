# backend/main.py
# Minimal RAG + Ollama backend for CareBot
import subprocess
import shlex
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uvicorn
from typing import Optional

app = FastAPI(title="CareBot (Ollama RAG)")

# --- CONFIG: choose the ollama model you tested (e.g., "phi3", "phi3-mini", "llama3") ---
OLLAMA_MODEL = "phi3"  # change if you used a different model

# --- Embedding model (small, fast) ---
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Small curated KB (replace/expand later with CBT snippets & templates) ---
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

class Message(BaseModel):
    text: str

def retrieve(user_text: str, k: int = 1):
    v = embed_model.encode([user_text], convert_to_numpy=True)
    D, I = index.search(v, k)
    return [KB[i] for i in I[0]]

def build_prompt(user_text: str, retrieved: str):
    # concise, guiding prompt for small models
    prompt = (
        "You are a calm, empathetic, and supportive assistant. "
        "Respond kindly, validate feelings, and offer a short coping suggestion. "
        "Do NOT provide medical advice. Keep responses to 1-3 short sentences.\n\n"
        f"Context: {retrieved}\n"
        f"User: {user_text}\n\n"
        "Assistant:"
    )
    return prompt

def query_ollama(prompt: str, timeout: int = 20) -> str:
    """
    Try a couple of common Ollama CLI forms to get a response.
    Returns the textual reply (best-effort) or raises a RuntimeError.
    """
    # Attempt 1: ollama generate (modern CLI)
    try:
        cmd = ["ollama", "generate", OLLAMA_MODEL, "--prompt", prompt]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode == 0 and p.stdout:
            text = p.stdout.decode("utf-8", errors="ignore").strip()
            # ollama generate may include the prompt; try to split safely
            # return the full output as best-effort
            return text
    except Exception:
        pass

    # Attempt 2: ollama run with input on stdin
    try:
        cmd = ["ollama", "run", OLLAMA_MODEL]
        p = subprocess.run(cmd, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode == 0 and p.stdout:
            text = p.stdout.decode("utf-8", errors="ignore").strip()
            return text
    except Exception:
        pass

    # Attempt 3: use shell style (fallback)
    try:
        shell_cmd = f"ollama run {shlex.quote(OLLAMA_MODEL)}"
        p = subprocess.run(shell_cmd, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, shell=True)
        if p.returncode == 0 and p.stdout:
            return p.stdout.decode("utf-8", errors="ignore").strip()
    except Exception:
        pass

    # If all attempts failed, provide helpful error message
    raise RuntimeError("Could not get a response from Ollama via CLI. Ensure `ollama` is on PATH and the model name is correct.")

@app.post("/chat")
def chat(msg: Message):
    user = (msg.text or "").strip()
    if not user:
        return {"reply": "Hey — I didn't catch that. Could you try saying that again?"}

    # --- simple rule-based crisis detection (first safety layer) ---
    crisis_keywords = [
        "kill myself", "suicide", "want to die", "end my life", "hurt myself",
        "i can't go on", "i'm going to kill myself", "i want to die"
    ]
    lowered = user.lower()
    if any(k in lowered for k in crisis_keywords):
        # Escalation response: do not call the model; provide emergency resources
        esc_text = (
            "I'm really sorry you're feeling this. If you're in immediate danger, please call your local emergency number now. "
            "If you can, consider contacting a suicide prevention hotline or a trusted person. You deserve help and care."
        )
        return {"reply": esc_text, "escalation": True}

    # --- RAG retrieval ---
    retrieved = retrieve(user, k=1)[0]

    # --- Build prompt and call Ollama ---
    prompt = build_prompt(user, retrieved)
    try:
        raw = query_ollama(prompt)
    except RuntimeError as e:
        # graceful fallback: echo empathy template + retrieved context
        fallback = (
            f"I hear you. {retrieved} "
            "If you can tell me a bit more, I'm listening."
        )
        return {"reply": fallback, "retrieved": retrieved, "warning": str(e)}

    # Try to extract reply: remove prompt if echoed
    if "Assistant:" in raw:
        reply = raw.split("Assistant:")[-1].strip()
    else:
        # if Ollama returns full text, try to remove the user and context lines (best-effort)
        reply = raw.strip()
        # keep reply length reasonable
        if len(reply) > 1000:
            reply = reply[:800] + "..."

    # final fallback if reply empties
    if not reply:
        reply = "Thanks for telling me that — I'm here with you. Would you like to say more?"

    return {"reply": reply, "retrieved": retrieved}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
