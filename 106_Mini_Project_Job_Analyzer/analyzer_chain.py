from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class JobData(BaseModel):
    name: str = Field(description="Name of the Candidate")
    total_experience: str = Field(description="Total year of experience")
    skills: List[str] = Field(description="List of skills")
    education: str = Field(description="Education details")
    certifications: Optional[List[str]] = Field(default=None, description="List of certifications")
    projects: Optional[List[str]] = Field(default=None, description="List of projects") 
    current_job_role: Optional[str] = Field(default=None, description="Current job role")

parser = PydanticOutputParser(pydantic_object=JobData)
format_instructions = parser.get_format_instructions()

job_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at reading and extracting data from Resume. Extract important data as structured JSON."),
    ("human", "Extract structured data from this job description:\n\n{resume_description}\n\n{format_instructions}")
])

llm = ChatOpenAI(model="gpt-4", temperature=0.0)

summary_prompt = PromptTemplate.from_template(
    "Summarize the key points of the following job description:\n\n{resume_description}"
   )

job_chain = job_prompt | llm | parser

def extract_job(resume_text: str) -> JobData:
    try:
        return job_chain.invoke({
            "resume_description": resume_text,
            "format_instructions": format_instructions
        })
    except Exception as e:
        print("❌ Error during job extraction:", e)
        return None 

def summarize_resume(resume_text: str) -> str:
    try:
        summary_chain = summary_prompt | llm
        return summary_chain.invoke({"resume_description": resume_text})
    except Exception as e:
        print("❌ Error during summarization:", e)
        return "Failed to summarize job description."