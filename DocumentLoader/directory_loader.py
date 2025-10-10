from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader= DirectoryLoader(
    path='documents',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

#docs = loader.load()
#print(len(docs))

docs = loader.lazy_load()

#print(docs)

for document in docs:
    print(document.metadata)