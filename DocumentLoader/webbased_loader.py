from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template='Answer the following questions \n {question} from the following \
        \n {text}',
    input_variables=['question','text']
)

parser= StrOutputParser()

url= 'https://www.flipkart.com/royal-enfield-classic-350-booking-ex-showroom-price/p/itmd8b7c092dffeb?pid=MCYHBHFMPAZUS3UQ&lid=LSTMCYHBHFMPAZUS3UQZQNO2Q&marketplace=FLIPKART&q=royal+enfield+bike&store=7dk%2F0aj&srno=s_1_5&otracker=AS_QueryStore_OrganicAutoSuggest_1_8_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_8_na_na_na&fm=organic&iid=f84a6a2f-dafc-408b-82fc-6a12ffaf8f61.MCYHBHFMPAZUS3UQ.SEARCH&ppt=hp&ppn=homepage&ssid=vkhao6l5680000001760076894296&qH=6ef7938c8daa4f33'
loader=WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)

chain = prompt | model | parser

res = chain.invoke({'question': input("Enter question : "), 'text' : docs[0].page_content})

print(res)