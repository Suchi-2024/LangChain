from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableMap

load_dotenv()

model1=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

llm= HuggingFaceEndpoint(
     repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)

model2=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Generate short and simple note from the following text \n {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="Merge the provided notes and Q&A into a single comprehensive summary:\nNotes:\n{notes}\n\nQ&A:\n{qa}",
    input_variables=["notes", "qa"]
)

parser=StrOutputParser()

parallel_chain= RunnableParallel({
    'notes': prompt1 | model2 | parser, 
    'qa': prompt2 | model1 | parser
})

# merge step
merge_chain = RunnableMap({
    "notes": lambda x: x["notes"],
    "qa": lambda x: x["qa"]
}) | prompt3 | model2 | parser

chain= parallel_chain | merge_chain

input_text = """
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position
wise fully connected feed-forward network. We employ a residual connection [11] around each of
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension dmodel = 512.
Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.
"""

result = chain.invoke({"text": input_text})
print(result)
 
chain.get_graph().print_ascii()