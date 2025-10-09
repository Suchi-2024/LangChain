from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.95)

prompt1= PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='Generate a LinkedIn post about : {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

parallel_chain= RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'ML'})

#print(result)

print(f"Tweet Post : {result["tweet"]}")
print(f"LinkedIn Post : {result['linkedin']}")