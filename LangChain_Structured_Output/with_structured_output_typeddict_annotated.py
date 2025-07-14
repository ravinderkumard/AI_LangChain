from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional,Literal

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "Summary of the review"]
    sentiment: Annotated[Literal["pos","neg"], "Return sentiment of the review, either neutral, positive or negative"]
    pros: Annotated[Optional[list[str]], "Write down all the pros of the review in a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons of the review in a list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" all good but water is leaking from AC inside. complimentary breakfast was good service was good and very near to the consulate The property was good it is very near to the consulate of the USA. But there was a leakage of water from the AC in the room which we had taken apart from that all was good. The complimentary breakfast was also good and neat 👍.

Travel Month: Jun 2025
Room: Deluxe King Bed""")

print("Summary:", result)
print("Sentiment:", result['sentiment'])
print("Structured Output:", result['summary']) 