from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8)

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Explain LLM and LangChain")
]

result=model.invoke(messages)

messages.append(AIMessage(result.content))

print(messages)