from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" The hardware is great, but the software is terrible. There are too many pre-installed apps that i can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update soon to fix these issues.""")

print("Summary:", result)
print("Sentiment:", result['sentiment'])
print("Structured Output:", result['summary']) 