from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()
prompt2 = PromptTemplate(
    template='Summarize the following text:\n\n{text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(
    prompt1 | llm | parser
)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(
        prompt2 | llm | parser
    )),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_gen_chain | branch_chain
)
result = final_chain.invoke({'topic': 'The Impact of AI on Modern Society'})
print("Generated Report:\n", result)

# The RunnableBranch allows us to create conditional branches in the workflow.
# This is useful for tasks where we want to perform different actions based on the output of a previous step.
# The RunnableBranch can be used to create conditional workflows that adapt based on the output of previous steps.
# The RunnableBranch can be used to create conditional workflows that adapt based on the output of previous steps.          