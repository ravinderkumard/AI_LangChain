from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke:\n\n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Explain the In French:\n\n{text}',
    input_variables=['text']
)

chain = RunnableSequence(
    prompt1 | llm | parser | prompt2 | llm | parser | prompt3 | llm | parser
)

result = chain.invoke({'topic': 'AI'})
print(result)