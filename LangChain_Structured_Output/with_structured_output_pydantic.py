from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional,Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="Summary of the review")
    sentiment: Literal["pos","neg"] = Field(description="Return sentiment of the review, either neutral, positive or negative")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros of the review in a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons of the review in a list")
    name: Optional[str] = Field(default=None, description="Name of the person giving the review")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" all good but water is leaking from AC inside. complimentary breakfast was good service was good and very near to the consulate The property was good it is very near to the consulate of the USA. But there was a leakage of water from the AC in the room which we had taken apart from that all was good. The complimentary breakfast was also good and neat 👍.

Travel Month: Jun 2025
Room: Deluxe King Bed
                                 Name: John Doe""")

print("Summary:", result)
print("Sentiment:", result.summary)
print("Structured Output:", result.sentiment) 