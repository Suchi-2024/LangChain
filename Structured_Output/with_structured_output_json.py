from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# Simplified schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neu"],  # Added "neu" for neutral
            "description": "Return sentiment of the review: positive, negative, or neutral"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)

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

print(result)