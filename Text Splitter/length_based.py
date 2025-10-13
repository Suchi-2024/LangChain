from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('ML.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(len(result))
print(result[1].page_content)