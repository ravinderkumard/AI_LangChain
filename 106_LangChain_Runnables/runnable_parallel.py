from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.0)
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a LinkedIn post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweet": prompt1 | llm | parser,
        "linkedin_post": prompt2 | llm | parser
    }
)
result = parallel_chain.invoke({"topic": "AI in 2024"})

print("Generated Content:\n")
print("-------------------\n")
print("Tweet:", result["tweet"])
print("\n-------------------")
print("\nLinkedIn Post:", result["linkedin_post"])