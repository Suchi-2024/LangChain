from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

prompt = PromptTemplate(
    template = 'Write a joke about {topic} in short',
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.95)

parser = StrOutputParser()

joke_gen_chain= RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x: len(x.split()))

})

final_chain= RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'LLM'})
print(f"Joke is : {result['joke']} and word count of the joke is : {result['word_count']}")