from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())

runnable_word_count = RunnableLambda(
    func=word_count
)

print(runnable_word_count.invoke("This is a test sentence."))

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(
    prompt1 | llm | parser 
)

parallel_chain = RunnableParallel(
    {
        "joke": joke_gen_chain,
        "word_count": RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_gen_chain | parallel_chain)
print(final_chain.invoke({'topic': 'AI'}))
# Output:
# {
#     "joke": "Why did the AI break up with its partner? Because it couldn't handle the emotional data!",
#     "word_count": 10
# }
# The joke is 10 words long.
# The RunnableLambda allows us to define custom functions that can be integrated into the LangChain workflow.
# This is useful for tasks like word counting, data validation, or any custom processing that needs to be done on the output of a chain.
# The RunnableLambda can be used anywhere in the chain, allowing for flexible and powerful workflows.
# The RunnableLambda can be used to create custom processing steps in a LangChain workflow.
# This allows for more complex and tailored workflows that can adapt to specific needs.
# The RunnableLambda can be used to create custom processing steps in a LangChain workflow.
# This allows for more complex and tailored workflows that can adapt to specific needs.
# The RunnableLambda can be used to create custom processing steps in a LangChain workflow.
# This allows for more complex and tailored workflows that can adapt to specific needs.         