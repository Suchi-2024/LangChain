from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person should be greater than 18")
    city: str = Field(description="City where the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate a random person with name, age, and city of a fictional {place} character in JSON format:\n{format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt = template.invoke({"place": "Indian"})

# print(prompt)

# result = model.invoke(prompt)

# Ensure the result is in JSON format
# try:
#     final_result = parser.parse(result.content)
#     print(final_result)
# except Exception as e:
#     print(f"Error parsing result: {e}")
#     print("Model output:", result.content)


chain= template | model | parser

final_result=chain.invoke({'place':'Indian'})

print(final_result)
