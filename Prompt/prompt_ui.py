from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.95)

st.header("Research Tool")

papers = ["Select a Paper", 
          "Attention Is All You Need", 
          "BERT: Pre-training of Deep Bidirectional Transformers", 
          "GPT-3: Language Models are Few-Shot Learners", 
          "Diffusion Models Beat GANs on Image Synthesis"]

styles = ["Select a Style", 
          "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]

lengths = ["Select Length", 
           "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]

paper_input = st.selectbox("Select Research Paper Name", papers)
style_input = st.selectbox("Select Explanation Style", styles)
length_input = st.selectbox("Select Explanation Length", lengths)

if paper_input == "Select a Paper" or style_input == "Select a Style" or length_input == "Select Length":
    st.warning("⚠️ Please make a valid selection in all dropdowns.")
else:
    pass

# Template
template=load_prompt('template.json')


submit=st.button("Summarize")

if submit:
    st.write("Generating summary...")
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)  