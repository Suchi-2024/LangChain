from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function  # Updated import
from typing import Optional, Literal, List

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# Schema with Pydantic
class Review(BaseModel):
    key_themes: List[str] = Field(..., description="Write down all the key themes discussed in the review")
    summary: str = Field(..., description="A brief summary of the review")
    sentiment: Literal["pos", "neg", "neu"] = Field(..., description="Return sentiment of the review: pos, neg, or neu")
    pros: Optional[List[str]] = Field(default_factory=list, description="Write down all the pros inside a list")
    cons: Optional[List[str]] = Field(default_factory=list, description="Write down all the cons inside a list")
    rating: Literal["1", "2", "3", "4", "5"] = Field(..., description="Rating out of 5")
    name: Optional[str] = Field(default="Anonymous", description="Name of the reviewer")

# Convert Pydantic schema to OpenAI function schema using the new function
review_schema = convert_to_openai_function(Review)

# Structured output
structured_model = model.with_structured_output(review_schema)

text = """I have been using the Dell Inspiron 15 3000 for over 3 years, and the experience has been disappointing. The look is attractive, but the performance is subpar.
From the very first year, I started facing hardware issues. The touchpad rarely works properly, and the display 
unexpectedly tore within just 6 months of purchase. The RAM is insufficient, lag is noticeable in daily tasks, 
and storage quickly became a major issue. Even after attempting upgrades, the performance did not improve much. 

The worst part is the battery life — it barely lasts 1 hour even with battery saver on, otherwise it drains even faster. 
This makes the laptop almost unusable for long sessions without being plugged in. 

Pros:
- Looks good
- Affordable price
- Keyboard is comfortable
- Decent for office work

Cons:
- Touchpad frequently fails
- Display tore within 6 months
- Insufficient RAM
- Frequent lag
- Storage limitations
- Upgrades do not improve performance
- Battery lasts only 1 hour
- Bad choice for memory intensive tasks
- Overheats quickly


Reviewed by Suchintika
"""

result = structured_model.invoke(text)

# Accessing specific fields
print(f"Key themes are : {result['key_themes']}")
print(f"Rating of the user : {result['rating']}")
print(f"Summary of the review : {result['summary']}")
print(f"Reviewer : {result['name']}")
