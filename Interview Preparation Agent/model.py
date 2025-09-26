from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from typing import Literal
import streamlit as st

load_dotenv()

#Heading
st.title("🤖 AI Interview Question Generator")
st.write(
    "Generate tailored interview questions based on a topic, difficulty level, and number of questions. "
    "Perfect for interview prep or practice sessions!"
)

# LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Pydantic parser
class InterviewQuestions(BaseModel):
    topic: str = Field(max_length=50, description="Enter the topic")
    number: int = Field(gt=0, lt=25, description="Number of interview questions")
    level: Literal["Easy", "Medium", "Hard"] = Field(description="Level of interview questions")
parser = PydanticOutputParser(pydantic_object=InterviewQuestions)

# Prompts
input_prompt = PromptTemplate(
    template="Extract topic, number, level from: topic={topic}, number={number}, level={level}\n{format_instructions}",
    input_variables=["topic", "number", "level"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

plan_prompt = PromptTemplate(
    template="Create a sequence of {number} interview questions for {level} level on the topic '{topic}'",
    input_variables=["topic", "number", "level"]
)

output_prompt = PromptTemplate(
    template="Format the following questions clearly:\n{planned_questions}",
    input_variables=["planned_questions"]
)

# RunnableSequence
agent_sequence = (
    {
        "topic": lambda x: x["topic"],
        "number": lambda x: x["number"],
        "level": lambda x: x["level"]
    } 
    | input_prompt 
    | model 
    | parser 
    | (
        lambda x: plan_prompt.format(
            topic=x.topic,
            number=x.number,
            level=x.level
        )
    )
    | model
    | (lambda x: output_prompt.format(planned_questions=x.text))
    | model
)



# Run agent
# topic = input("Enter a topic: ")
# number = int(input("Enter number of questions: "))
# level = input("Enter level of questions (Easy, Medium, Hard): ")

# print("\nGenerated Interview Questions:\n")
# print(result.content)

# Inputs
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
    
    # Show AI response
    st.chat_message("assistant").write(ai_response.content)
else:
    st.warning("Enter a topic and click submit to proceed")