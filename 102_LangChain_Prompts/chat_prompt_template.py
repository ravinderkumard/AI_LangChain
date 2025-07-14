from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant in the domain of {domain}. '),
    ('human', 'Discuss the topic of {topic} in detail, providing insights and examples. ')
])

prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'swing bowling'
})

print(prompt)  # Display the content of the system message