from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.0)

schema = [
    ResponseSchema(name='fact_1',description="Fact 1 about the topic"),
    ResponseSchema(name='fact_2',description="Fact 2 about the topic"),
    ResponseSchema(name='fact_3',description="Fact 3 about the topic")
]

#Define parser
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give me 3 fact about topic:{topic}\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)



# prompt = template.invoke({'topic':'black hole'})

# result = llm.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | llm | parser

result = chain.invoke({'topic','black hole'})

print(result)
