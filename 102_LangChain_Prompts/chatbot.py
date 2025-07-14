from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

chat_history = [
    SystemMessage(content="You are a helpful assistant. Please answer the user's questions to the best of your ability.")
]

while True:
    user_input = input("Enter your prompt (or 'exit' to quit)! You: ")
    #
    # chat_history.append({"role": "user", "content": user_input})
    chat_history.append(HumanMessage(content=user_input))  # Append user input as a HumanMessage
    if user_input.lower() == 'exit':
        break
    try:
        result = model.invoke(chat_history)
        #chat_history.append({"role": "assistant", "content": result.content})
        chat_history.append(AIMessage(content=result.content))  # Append AI response as an AIMessage
        print("AI:", result.content)  # Display the content of the response
    except Exception as e:
        print(f"An error occurred: {e}")

print("Chat ended.",chat_history)