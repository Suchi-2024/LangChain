from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["HF_HOME"] = "E:/3rd Sem/huggingface_cache"
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ,
    task="text-generation",
    pipeline_kwargs={
        "max_length":512,
        "max_new_tokens":100,
        "temperature":0.5
        }
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of India?")

print("Raw result:", result)
try:
    print("AI reply:", result.content)
except AttributeError:
    pass


model_id = "TheBloke/TinyLlama-400M-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",   # maps model to GPU if available
    load_in_8bit=True    # reduces memory footprint drastically
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("What is the capital of India?", max_new_tokens=50))
