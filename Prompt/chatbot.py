from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.95)

chat_history=[
    SystemMessage(content="You are a helpful assistant")
]

while True:
    user_input=input("You : ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print(f"AI  : {result.content}")

print(chat_history)