from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


load_dotenv()

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("top News in india today")

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, [search_tool],prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

response = agent_executor.invoke({"input": "Countries in Asia and their capitals names?"})

print(response)

print(response['output'])