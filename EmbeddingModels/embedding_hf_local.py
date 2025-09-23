from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#text="Delhi is the capiatal of India."

#vector=embedding.embed_query(text)

documents=["Delhi is the capital of India.",
           "Paris is the capital of France.",
           "Kolkata is the capital of West Bengal."]

vector=embedding.embed_documents(documents)
print(str(vector))