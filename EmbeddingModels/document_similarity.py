from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.",
    "Machine Learning (ML) is a subset of AI that enables systems to learn and improve from data without explicit programming.",
    "Data Science involves extracting knowledge and insights from structured and unstructured data using statistical, mathematical, and computational techniques.",
    "Deep Learning is a specialized branch of ML that uses neural networks with many layers to model complex patterns in data.",
    "Natural Language Processing (NLP) is a field of AI that focuses on enabling machines to understand, interpret, and generate human language."
]

query = input("Ask something : ")

doc_embeddings= embeddings.embed_documents(documents)
query_embedding= embeddings.embed_query(query)

scores=cosine_similarity([query_embedding], doc_embeddings)[0]

index,score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Ans of your query : {query}")
print(documents[index])
print("Similarity Score is : ",score)
