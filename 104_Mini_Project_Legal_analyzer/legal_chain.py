from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Define output structure using Pydantic
class Clause(BaseModel):
    title: str = Field(description="Title of the clause")
    content: str = Field(description="Content of the clause")

class ContractAnalysis(BaseModel):
    clauses: List[Clause]

# ✅ Output parser
parser = PydanticOutputParser(pydantic_object=ContractAnalysis)
format_instructions = parser.get_format_instructions()

# ✅ Prompt for extracting clauses
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a legal expert specializing in contract analysis. Extract important clauses as structured JSON."),
    ("human", "Extract clauses from this contract:\n\n{contract_text}\n\n{format_instructions}")
])

# ✅ LLM setup
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# ✅ Chain definition (MUST come after parser, prompt, llm)
chain = prompt | llm | parser  # 🔁 This must be defined globally before any function uses it

# ✅ Prompt for summarization
summary_prompt = PromptTemplate.from_template(
    "Summarize the key points of the following contract:\n\n{contract_text}"
)

# ✅ Clause extraction function
def extract_contract_clauses(contract_text: str) -> Optional[ContractAnalysis]:
    try:
        return chain.invoke({
            "contract_text": contract_text,
            "format_instructions": format_instructions
        })
    except Exception as e:
        print("❌ Error during clause extraction:", e)
        return None

# ✅ Contract summarization function
def summarize_contract(contract_text: str) -> str:
    try:
        summary_chain = summary_prompt | llm
        return summary_chain.invoke({"contract_text": contract_text})
    except Exception as e:
        print("❌ Error during summarization:", e)
        return "Failed to summarize contract."
