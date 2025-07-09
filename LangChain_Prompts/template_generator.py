from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="Explain the paper '{paper_input}' in a {style_input} style with a {length_input} length."
)

template.save("paper_explanation_template.json");