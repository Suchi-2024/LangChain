from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description = "Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description = "Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description = "Fact 3 about the topic"),
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="Give 3 fact about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

final_result=chain.invoke({'topic':'Relativity Theory'})

# prompt= template.invoke({'topic':'Relativity Theory'})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

print(final_result)