from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize the OpenAI chat model
llm = ChatOpenAI(model="gpt-4", temperature=0.0)
os.environ["USER_AGENT"] = "my-langchain-bot/1.0"
prompt = PromptTemplate(
    template="Answer the following question:\n\n{question} from following test: \n\n{text}",
    input_variables=["question","text"]
)

parser = StrOutputParser()


urls = "https://medium.com/@narasimha4789/integrate-hashicorp-vault-in-spring-boot-application-to-read-application-secrets-using-docker-aa52b417f484"
loader = WebBaseLoader(urls)
documents = loader.load()
print(documents[0].page_content[:1000])  # Print first 1000 characters of content for verification

chain = prompt | llm | parser
result = chain.invoke({'text': documents[0].page_content, 'question': 'What is HashiCorp Vault?'})
print(result)