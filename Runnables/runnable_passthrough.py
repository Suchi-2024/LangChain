from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.95)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "Here is the joke : {text} and Explanantion of  it : {text} in short",
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result=final_chain.invoke("Love")

#print(result)

print(result['joke'])
print()
print(result['explanation'])