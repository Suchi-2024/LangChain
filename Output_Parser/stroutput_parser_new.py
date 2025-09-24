from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

# 1st Prompt

template1= PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Write a 2 line summary of the following text:\n {text}",
    input_variables=['text']
)

parser= StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result= chain.invoke({'topic':'Relativity Theory'})

print(result)