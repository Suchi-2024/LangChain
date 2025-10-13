from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import Literal
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Load environment ---
load_dotenv()

# --- Initialize LLM ---
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)

# --- Initialize session state ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_session_index" not in st.session_state:
    st.session_state.current_session_index = None

# --- App Title ---
st.title("🤖 AI Interview Question Generator & Multi-Session Chat")

# --- Sidebar (Chat Management) ---
with st.sidebar:
    st.header("💬 Manage Chats")
    if st.button("➕ New Chat"):
        st.session_state.chat_sessions.append({
            "chat_history": [SystemMessage(content="You are an Interview Preparation Agent.")],
            "generated_questions": [],
            "topic": None  # store topic explicitly
        })
        st.session_state.current_session_index = len(st.session_state.chat_sessions) - 1
        st.rerun()

    if st.session_state.chat_sessions:
        session_titles = [f"Session {i+1}" for i in range(len(st.session_state.chat_sessions))]
        selected_index = st.selectbox(
            "Select Chat Session",
            options=range(len(session_titles)),
            format_func=lambda x: session_titles[x],
            index=st.session_state.current_session_index or 0
        )
        st.session_state.current_session_index = selected_index

# --- If no sessions exist, show Welcome Page ---
if not st.session_state.chat_sessions:
    st.markdown("""
        ### 👋 Welcome!
        Start preparing for your interviews by generating AI-powered questions 
        and chatting with your personal assistant.

        👉 Click below to **start your first chat session**:
    """)
    if st.button("🚀 Start Chat"):
        st.session_state.chat_sessions.append({
            "chat_history": [SystemMessage(content="You are an Interview Preparation Agent.")],
            "generated_questions": [],
            "topic": None
        })
        st.session_state.current_session_index = 0
        st.rerun()
    st.stop()

# --- Active Session ---
current_session = st.session_state.chat_sessions[st.session_state.current_session_index]

# --- Question Generator UI ---
st.subheader("🎯 Generate Interview Questions")
session_key = st.session_state.current_session_index

topic = st.text_input("Topic", placeholder="e.g., Machine Learning", key=f"topic_{session_key}")
number = st.number_input("Number of questions", min_value=1, max_value=20, value=5, key=f"number_{session_key}")
level = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key=f"level_{session_key}")
submit = st.button("Generate Questions", key=f"submit_{session_key}")

# --- Schema for structured output ---
class FinalInterviewOutput(BaseModel):
    topic: str
    number: int
    level: Literal["Easy", "Medium", "Hard"]
    qa_pairs: list[dict]  # Each item: {"question": str, "answer": str}

final_parser = PydanticOutputParser(pydantic_object=FinalInterviewOutput)

# --- Prompt Template ---
final_prompt = PromptTemplate(
    template=(
        "Generate {number} interview questions with concise, high-quality answers "
        "on the topic '{topic}' at {level} difficulty level.\n\n"
        "Return the output as a JSON object with this structure:\n"
        "{format_instructions}"
    ),
    input_variables=["topic", "number", "level"],
    partial_variables={"format_instructions": final_parser.get_format_instructions()}
)

# --- Generate Interview Questions ---
if submit:
    st.chat_message("user").write(f"Topic: {topic}, Number: {number}, Level: {level}")

    if not topic.strip():
        warning_msg = "⚠️ Please enter a valid topic to generate questions."
        st.chat_message("assistant").write(warning_msg)
        current_session["chat_history"].append(AIMessage(content=warning_msg))
    else:
        with st.spinner("Generating interview questions and answers..."):
            agent_sequence = final_prompt | model | final_parser
            ai_response = agent_sequence.invoke({
                "topic": topic,
                "number": number,
                "level": level
            })

        current_session["generated_questions"] = ai_response.qa_pairs
        current_session["topic"] = topic  # store topic for chat enforcement

         # --- Append new questions instead of overwriting ---
        existing_count = len(current_session["generated_questions"])
        for i, pair in enumerate(ai_response.qa_pairs):
            current_session["generated_questions"].append(pair)
            qa_text = f"Q{existing_count + i + 1}. {pair['question']}\nA{existing_count + i + 1}. {pair['answer']}"
            current_session["chat_history"].append(AIMessage(content=qa_text))
            
        # Format Q&A nicely
        for i, pair in enumerate(ai_response.qa_pairs):
            qa_text = f"Q{i+1}. {pair['question']}\nA{i+1}. {pair['answer']}"
            current_session["chat_history"].append(AIMessage(content=qa_text))

# --- Display Q&A and topic-focused chat ---
if current_session["generated_questions"]:
    st.subheader("📝 Generated Questions & Answers")
    
    for i, pair in enumerate(current_session["generated_questions"]):
        with st.expander(f"Q{i+1}: {pair['question']}"):
            st.markdown(f"**Answer:**\n\n{pair['answer']}")

    st.divider()
    st.markdown("### 💬 Continue Chat with the topic")
    user_input = st.chat_input("Ask about the questions, get hints, or explanations...")

    if user_input:
        st.chat_message("user").write(user_input)
        current_session["chat_history"].append(HumanMessage(content=user_input))

        # --- Check if input is relevant to topic ---
        relevance_prompt = f"""
        The current interview topic is: "{current_session['topic']}".
        The user message is: "{user_input}".
        Is this message relevant to the topic (answer yes or no only)?
        """
        relevance_check = model.invoke(relevance_prompt).content.strip().lower()

        if "no" in relevance_check:
            warning_text = (
                f"⚠️ Your message doesn't seem related to the topic "
                f"'{current_session['topic']}'. Please stay on topic, "
                "or start a new chat from the sidebar."
            )
            st.chat_message("assistant").warning(warning_text)
            current_session["chat_history"].append(AIMessage(content=warning_text))
            st.stop()

        # --- Continue topic-based chat ---
        qa_context = "\n\n".join([
            f"Q{i+1}: {pair['question']}\nA{i+1}: {pair['answer']}"
            for i, pair in enumerate(current_session["generated_questions"])
        ])

        context_prompt = (
            f"You are an Interview Preparation Assistant specialized in '{current_session['topic']}'.\n\n"
            f"Here are the generated questions and answers:\n{qa_context}\n\n"
            f"The user says: '{user_input}'.\n"
            f"Respond concisely (~100 words) and stay strictly within the topic."
        )

        with st.spinner("AI is thinking..."):
            ai_result = model.invoke(context_prompt)

        st.chat_message("assistant").write(ai_result.content)
        current_session["chat_history"].append(AIMessage(content=ai_result.content))
else:
    st.info("👆 Please generate interview questions first to start chatting.")
