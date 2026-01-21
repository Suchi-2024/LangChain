import os
import streamlit as st
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --------------------------------------------------
# Environment & API Key (Streamlit-safe)
# --------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# --------------------------------------------------
# Initialize Gemini LLM (EXPLICIT api_key)
# --------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    api_key=GOOGLE_API_KEY
)

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_session_index" not in st.session_state:
    st.session_state.current_session_index = None

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("🤖 AI Interview Question Generator & Multi-Session Chat")

# --------------------------------------------------
# Sidebar – Chat Management
# --------------------------------------------------
with st.sidebar:
    st.header("💬 Manage Chats")

    if st.button("➕ New Chat"):
        st.session_state.chat_sessions.append({
            "chat_history": [
                SystemMessage(content="You are an Interview Preparation Agent.")
            ],
            "generated_questions": [],
            "topic": None
        })
        st.session_state.current_session_index = len(
            st.session_state.chat_sessions
        ) - 1
        st.rerun()

    if st.session_state.chat_sessions:
        session_titles = [
            f"Session {i + 1}"
            for i in range(len(st.session_state.chat_sessions))
        ]
        selected_index = st.selectbox(
            "Select Chat Session",
            options=range(len(session_titles)),
            format_func=lambda x: session_titles[x],
            index=st.session_state.current_session_index or 0
        )
        st.session_state.current_session_index = selected_index

# --------------------------------------------------
# Welcome Screen
# --------------------------------------------------
if not st.session_state.chat_sessions:
    st.markdown("""
        ### 👋 Welcome!
        Start preparing for your interviews by generating AI-powered questions 
        and chatting with your personal assistant.
    """)
    if st.button("🚀 Start Chat"):
        st.session_state.chat_sessions.append({
            "chat_history": [
                SystemMessage(content="You are an Interview Preparation Agent.")
            ],
            "generated_questions": [],
            "topic": None
        })
        st.session_state.current_session_index = 0
        st.rerun()
    st.stop()

# --------------------------------------------------
# Active Session Safety
# --------------------------------------------------
if st.session_state.current_session_index is None:
    st.stop()

current_session = st.session_state.chat_sessions[
    st.session_state.current_session_index
]

# --------------------------------------------------
# Question Generator UI
# --------------------------------------------------
st.subheader("🎯 Generate Interview Questions")

session_key = st.session_state.current_session_index

topic = st.text_input(
    "Topic",
    placeholder="e.g., Machine Learning",
    key=f"topic_{session_key}"
)

number = st.number_input(
    "Number of questions",
    min_value=1,
    max_value=20,
    value=5,
    key=f"number_{session_key}"
)

level = st.selectbox(
    "Difficulty",
    ["Easy", "Medium", "Hard"],
    key=f"level_{session_key}"
)

submit = st.button("Generate Questions", key=f"submit_{session_key}")

# --------------------------------------------------
# Output Schema
# --------------------------------------------------
class FinalInterviewOutput(BaseModel):
    topic: str
    number: int
    level: Literal["Easy", "Medium", "Hard"]
    qa_pairs: list[dict]

final_parser = PydanticOutputParser(
    pydantic_object=FinalInterviewOutput
)

# --------------------------------------------------
# Prompt Template
# --------------------------------------------------
final_prompt = PromptTemplate(
    template=(
        "Generate {number} interview questions with concise, high-quality answers "
        "on the topic '{topic}' at {level} difficulty level.\n\n"
        "Return the output strictly as JSON:\n"
        "{format_instructions}"
    ),
    input_variables=["topic", "number", "level"],
    partial_variables={
        "format_instructions": final_parser.get_format_instructions()
    }
)

# --------------------------------------------------
# Generate Interview Questions (NO DUPLICATION)
# --------------------------------------------------
if submit:
    st.chat_message("user").write(
        f"Topic: {topic}, Number: {number}, Level: {level}"
    )

    if not topic.strip():
        msg = "⚠️ Please enter a valid topic."
        st.chat_message("assistant").write(msg)
        current_session["chat_history"].append(
            AIMessage(content=msg)
        )
    else:
        with st.spinner("Generating interview questions..."):
            chain = final_prompt | model | final_parser
            ai_response = chain.invoke({
                "topic": topic,
                "number": number,
                "level": level
            })

        # OVERWRITE (no duplicates)
        current_session["generated_questions"] = ai_response.qa_pairs
        current_session["topic"] = topic

        for i, pair in enumerate(ai_response.qa_pairs):
            qa_text = (
                f"Q{i + 1}. {pair['question']}\n"
                f"A{i + 1}. {pair['answer']}"
            )
            current_session["chat_history"].append(
                AIMessage(content=qa_text)
            )

# --------------------------------------------------
# Display Q&A + Topic-focused Chat
# --------------------------------------------------
if current_session["generated_questions"]:
    st.subheader("📝 Generated Questions & Answers")

    for i, pair in enumerate(current_session["generated_questions"]):
        with st.expander(f"Q{i + 1}: {pair['question']}"):
            st.markdown(f"**Answer:**\n\n{pair['answer']}")

    st.divider()
    st.markdown("### 💬 Continue Chat (Topic-Focused)")

    user_input = st.chat_input(
        "Ask about the questions, get hints, or explanations..."
    )

    if user_input:
        st.chat_message("user").write(user_input)
        current_session["chat_history"].append(
            HumanMessage(content=user_input)
        )

        # Lightweight relevance check
        relevance_prompt = (
            f"Topic: {current_session['topic']}\n"
            f"Message: {user_input}\n"
            "Reply only YES or NO."
        )

        relevance_check = model.invoke(
            relevance_prompt
        ).content.strip().upper()

        if relevance_check != "YES":
            warning = (
                f"⚠️ Please stay on the topic "
                f"'{current_session['topic']}' "
                "or start a new chat."
            )
            st.chat_message("assistant").warning(warning)
            current_session["chat_history"].append(
                AIMessage(content=warning)
            )
            st.stop()

        qa_context = "\n\n".join([
            f"Q{i + 1}: {pair['question']}\n"
            f"A{i + 1}: {pair['answer']}"
            for i, pair in enumerate(
                current_session["generated_questions"]
            )
        ])

        context_prompt = (
            f"You are an Interview Preparation Assistant "
            f"specialized in '{current_session['topic']}'.\n\n"
            f"{qa_context}\n\n"
            f"User: {user_input}\n"
            "Respond concisely (~100 words)."
        )

        with st.spinner("AI is thinking..."):
            ai_reply = model.invoke(context_prompt)

        st.chat_message("assistant").write(ai_reply.content)
        current_session["chat_history"].append(
            AIMessage(content=ai_reply.content)
        )
else:
    st.info("👆 Generate interview questions to start chatting.")
