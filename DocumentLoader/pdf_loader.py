from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('bigdata.pdf')

data = loader.load()

print(len(data))

for i in range(len(data)):
    print(f"Content of page {i+1} is : ")
    print(data[i].page_content)