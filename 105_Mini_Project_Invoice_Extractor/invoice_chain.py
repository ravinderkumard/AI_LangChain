from pydantic import BaseModel, Field
from typing import List, Optional
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Define output structure using Pydantic
class LineItem(BaseModel):
    description: str = Field(description="Description of the line item")
    quantity: str = Field(description="Quantity of the item")
    unit_price: str = Field(description="Unit price of the item")
    total: str = Field(description="Total price for the line item")

class InvoiceData(BaseModel):
    vendor:str = Field(description="Vendor name")
    invoice_number: str = Field(description="Invoice number")
    invoice_date: str = Field(description="Date of the invoice")
    total_amount: str = Field(description="Total amount of the invoice")
    line_items: List[LineItem] = Field(description="List of line items in the invoice")

# ✅ Output parser
parser = PydanticOutputParser(pydantic_object=InvoiceData)
format_instructions = parser.get_format_instructions()

# ✅ Prompt for extracting
invoice_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at reading and extracting data from invoices. Extract important data as structured JSON."),
    ("human", "Extract structured data from this invoice:\n\n{invoice_text}\n\n{format_instructions}")
])

# ✅ LLM setup
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# ✅ Chain definition (MUST come after parser, prompt, llm)
invoice_chain = invoice_prompt | llm | parser  # 🔁 This must be defined globally before any function uses it

# ✅ Invoice extraction function
def extract_invoice(invoice_text: str) -> InvoiceData:
    try:
        return invoice_chain.invoke({
            "invoice_text": invoice_text,
            "format_instructions": format_instructions
        })
    except Exception as e:
        print("❌ Error during clause extraction:", e)
        return None


