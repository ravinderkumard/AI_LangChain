from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header("LangChain Prompt UI")
user_input = st.write("Enter your prompt below and click 'Submit' to get a response from the model.")
prompt = st.text_input("Prompt", "What is the capital of France?")
if st.button("Submit"):
    model = ChatOpenAI(model="gpt-4")
    result = model.invoke(prompt)
    st.write("Response from the model:")
    st.write(result.content)  # Display the content of the response