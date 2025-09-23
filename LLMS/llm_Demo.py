from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0.9,model="gpt-3.5-turbo-instruct")

result= llm.invoke("Tell me a joke about programming.")

print(result)
