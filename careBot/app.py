# streamlit_client.py
import streamlit as st
import requests


st.set_page_config(page_title="CareSpace", page_icon="🌿", layout="centered")
API = "http://127.0.0.1:8000/chat"

st.title("CareBot — There will be always someone who cares about you 🌿")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("f", clear_on_submit=True):
    text = st.text_input("You:")
    send = st.form_submit_button("Send")

if send and text:
    st.session_state.messages.append({"from": "user", "text": text})
    try:
        r = requests.post(API, json={"text": text}, timeout=10)
        data = r.json()
        bot = data.get("reply", "[no reply]")
        if data.get("escalation"):
            bot = "⚠️ " + bot  # highlight escalation
        st.session_state.messages.append({"from": "bot", "text": bot})
    except Exception as e:
        st.session_state.messages.append({"from": "bot", "text": f"[error contacting backend: {e}]"})


for m in st.session_state.messages:
    who = "You" if m["from"] == "user" else "CareBot"
    st.markdown(f"**{who}:** {m['text']}")
