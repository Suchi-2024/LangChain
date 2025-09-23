from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.95)
result=model.invoke("Tell me a joke about programming.")

#print(result)
print(result.content)