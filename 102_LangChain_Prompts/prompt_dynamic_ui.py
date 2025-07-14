from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatOpenAI(model="gpt-4")   # Initialize the model

st.header("Dynamic UI with LangChain and Streamlit")
paper_input = st.selectbox(
    "Select a paper",
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", 
     "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select explanation style", ["Simple", "Detailed", "Technical", "Mathematical"])

length_input = st.selectbox("Select length of explanation", ["Short", "Medium", "Long"])

template = load_prompt("paper_explanation_template.json")  # Load the prompt template



if st.button("Generate Explanation"):
    prompt = template.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    result = model.invoke(prompt)
    st.write("Explanation:")
    st.write(result.content)  # Display the content of the response