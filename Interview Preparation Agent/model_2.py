from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Initialize LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)

# Heading
st.title("🤖 AI Interview Question Generator & Multi-Session Chat")

# Intro Section
if not st.session_state.chat_sessions:
    st.markdown(
        """
        ### 👋 Welcome!
        Start preparing for your interviews by generating AI-powered questions 
        and chatting with your personal assistant.

        👉 Click below to **start your first chat session**:
        """
    )

    # Big button in the main page
    if st.button("🚀 Start Chat"):
        st.session_state.chat_sessions.append({
            "chat_history": [SystemMessage(content="You are an Interview Preparation Agent.")],
            "generated_questions": []
        })
        st.session_state.current_session_index = len(st.session_state.chat_sessions) - 1
        st.rerun()   # reload to jump directly into chat


# Sidebar: New Chat / Session selection
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_session_index" not in st.session_state:
    st.session_state.current_session_index = None

# Button to create new chat session
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_sessions.append({
        "chat_history": [SystemMessage(content="You are an Interview Preparation Agent.")],
        "generated_questions": []
    })
    st.session_state.current_session_index = len(st.session_state.chat_sessions) - 1

# Select existing session
if st.session_state.chat_sessions:
    session_titles = [f"Session {i+1}" for i in range(len(st.session_state.chat_sessions))]
    selected_index = st.sidebar.selectbox(
        "Select Chat Session",
        options=range(len(session_titles)),
        format_func=lambda x: session_titles[x],
        index=st.session_state.current_session_index or 0
    )
    st.session_state.current_session_index = selected_index
    current_session = st.session_state.chat_sessions[st.session_state.current_session_index]
else:
    st.warning("Create a new chat session to start.")
    st.stop()

# Sidebar: Generate Interview Questions
st.sidebar.header("Generate Interview Questions")
topic = st.sidebar.text_input("Topic", placeholder="e.g., Machine Learning")
number = st.sidebar.number_input("Number of questions", min_value=1, max_value=20, value=5)
level = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
submit = st.sidebar.button("Generate Questions")

# Schema for structured output
class FinalInterviewOutput(BaseModel):
    topic: str
    number: int
    level: Literal["Easy", "Medium", "Hard"]
    questions: list[str]

final_parser = PydanticOutputParser(pydantic_object=FinalInterviewOutput)

# Prompt template
final_prompt = PromptTemplate(
    template=(
        "Generate {number} interview questions on the topic '{topic}' at {level} level.\n\n"
        "Return the output as a JSON object with this structure:\n"
        "{format_instructions}"
    ),
    input_variables=["topic", "number", "level"],
    partial_variables={"format_instructions": final_parser.get_format_instructions()}
)

# Generate interview questions
if submit:
    st.chat_message("user").write(f"Topic: {topic}, Number: {number}, Level: {level}")

    if not topic.strip():  # Check for blank topic
        warning_msg = "⚠️ Please enter a valid topic to generate questions."
        st.chat_message("ai").write(warning_msg)
        current_session["chat_history"].append(AIMessage(content=warning_msg))
    else:
        with st.spinner("Generating interview questions..."):
            agent_sequence = final_prompt | model | final_parser
            ai_response = agent_sequence.invoke({
                "topic": topic,
                "number": number,
                "level": level
            })

        # Store questions in current session
        current_session["generated_questions"] = ai_response.questions

        # Display questions
        questions_text = "\n\n".join([f"Q{i+1}. {q}" for i, q in enumerate(ai_response.questions)])
        current_session["chat_history"].append(
            AIMessage(content=f"Here are your {ai_response.number} questions:\n\n{questions_text}")
        )

# Display chat messages for current session
for msg in current_session["chat_history"]:
    if isinstance(msg, dict):
        content = msg.get("content", "")
        role = msg.get("role", "ai")
    else:
        content = msg.content
        role = "user" if isinstance(msg, HumanMessage) else "ai"
    st.chat_message(role).write(content)

# Chat input for dynamic conversation
user_input = st.chat_input("Ask about the questions, get hints, or answers...")
if user_input:
    st.chat_message("user").write(user_input)
    current_session["chat_history"].append(HumanMessage(content=user_input))

    # Build context including generated questions
    context_prompt = (
        "Here are the generated interview questions:\n" +
        "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(current_session["generated_questions"])]) +
        f"\n\nUser asks: {user_input}\nPlease answer based on these questions."
    )

    with st.spinner("AI is typing..."):
        ai_result = model.invoke(context_prompt)

    st.chat_message("ai").write(ai_result.content)
    current_session["chat_history"].append(AIMessage(content=ai_result.content))
