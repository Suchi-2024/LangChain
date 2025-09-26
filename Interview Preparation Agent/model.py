from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import streamlit as st

load_dotenv()

# Heading
st.title("🤖 AI Interview Question Generator")
st.write(
    "Generate tailored interview questions based on a topic, difficulty level, and number of questions. "
    "Perfect for interview prep or practice sessions!"
)

# LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)

# Final schema for structured output
class FinalInterviewOutput(BaseModel):
    topic: str = Field(description="Interview topic")
    number: int = Field(description="Number of questions")
    level: Literal["Easy", "Medium", "Hard"] = Field(description="Difficulty level")
    questions: list[str] = Field(description="List of generated interview questions")

final_parser = PydanticOutputParser(pydantic_object=FinalInterviewOutput)

# Prompt (only one call to model)
final_prompt = PromptTemplate(
    template=(
        "Generate {number} interview questions on the topic '{topic}' at {level} level.\n\n"
        "Return the output as a JSON object with this structure:\n"
        "{format_instructions}"
    ),
    input_variables=["topic", "number", "level"],
    partial_variables={"format_instructions": final_parser.get_format_instructions()}
)

# RunnableSequence (optimized: only one LLM call)
agent_sequence = final_prompt | model | final_parser

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    topic = st.text_input("Enter the topic", placeholder="e.g., Machine Learning")
    number = st.number_input(
        label="Number of questions",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Select how many interview questions you want"
    )
    level = st.selectbox(
        label="Select difficulty level",
        options=["Easy", "Medium", "Hard"],
        help="Choose the difficulty of the questions"
    )
    submit = st.button("Generate Questions")

# Main area: generate output
if submit and topic:
    # Show user input
    st.chat_message("user").write(f"Topic: {topic}, Number: {number}, Level: {level}")
    
    # Loading spinner while generating
    with st.spinner("Generating interview questions..."):
        ai_response = agent_sequence.invoke({
            "topic": topic,
            "number": number,
            "level": level
        })
    
    # Show Heading
    st.subheader(f"📌 {ai_response.topic} — {ai_response.level} ({ai_response.number} Questions)")
    
    # Show Questions
    for i, q in enumerate(ai_response.questions, start=1):
        st.markdown(f"**Q{i}.** {q}")
else:
    st.warning("Enter a topic and click 'Generate Questions' to proceed")
