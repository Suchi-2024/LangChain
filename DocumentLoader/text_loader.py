from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model= ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.6)

prompt= PromptTemplate(
    template='Write a short summary for the following poem : {poem}',
    input_variables=['poem']
)

parser= StrOutputParser()

loader= TextLoader('Rag.txt', encoding='utf-8')

docs = loader.load()

#print(docs)

#print(type(docs))

print(type(docs[0]))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))