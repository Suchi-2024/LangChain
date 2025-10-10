from langchain_community.document_loaders import CSVLoader

loader=CSVLoader('liver_patient.csv')

docs = loader.load()

print(f"Length of the csv document is : {len(docs)}")

print(docs[0].page_content)