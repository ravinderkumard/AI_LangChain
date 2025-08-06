from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

passthrough = RunnablePassthrough()

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


joke_gen_chain = RunnableSequence(
    prompt1 | llm | parser 
)

parallel_chain = RunnableParallel(
    {
        "joke": joke_gen_chain,
        "explanation": prompt2 | llm | parser,
    }
)

final_chain = RunnableSequence(joke_gen_chain | parallel_chain)

print(final_chain.invoke({'topic': 'AI'}))