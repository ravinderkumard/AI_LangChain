from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Create the chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Customer Support Agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

# Prepare chat history (example)
chat_history = []

with open("chat_history.txt", "r") as file:
    for line in file:
        line = line.strip()
        if line.startswith("Human:"):
            chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
        elif line.startswith("AI:"):
            chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))


# Create the prompt with variables
formatted_prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

# Print the result
print(formatted_prompt)
