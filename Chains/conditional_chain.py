from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature =0.2)

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive","Negative","Neutral"]= Field(description= "Give sentiment of the feedback")

parser2= PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative or neutral \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1| model | parser2

prompt2= PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feeback}",
    input_variables=["feedback"]
)

prompt3= PromptTemplate(
    template="Write an appropriate response to this negaitive feedback \n {feeback}",
    input_variables=["feedback"]
)

prompt4= PromptTemplate(
    template="Write an appropriate response to this neutral feedback \n {feeback}",
    input_variables=["feedback"]
)

branch_chain= RunnableBranch(
    (lambda x: x.sentiment=='Postivie', prompt2 | model | parser),
    (lambda x: x.sentiment=='Negative', prompt3 | model |parser),
    (lambda x: x.sentiment== "Neutral", prompt4 | model |parser),
    RunnableLambda(lambda x: "Not able to identify the sentiment")
)

chain= classifier_chain | branch_chain

result= chain.invoke({"feedback": input("Enter your feedback : ")})

print(result)

chain.get_graph().print_ascii()