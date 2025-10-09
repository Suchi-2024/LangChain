from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a report on the {topic} ',
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.95)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Summarize the following text : {text}',
    input_variables=['topic']
)

report_gen_chain= RunnableSequence(prompt1, model, parser)

branch_chain= RunnableBranch(
    (lambda x: len(x.split())>200, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

res= final_chain.invoke('Current geopolitical problems between USA and India')

print(res)