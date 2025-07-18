from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template='Generate short and simple note from following text\n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question and answer from following text\n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document\n nodes->{notes} and quiz->{quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
Welcome to the next era of agile software development — agent-powered experimentation. The rise of agentic AI and LLMs (Large Language Models) isn’t just changing the tools we use; it’s changing the team itself.

From Sprints to Seconds
Agile teams thrive on iteration — the faster you can ship, the faster you can learn. But there’s a catch: iteration requires time, people, and coordination. Today, that equation changes.

Agentic AI — intelligent, goal-oriented agents that plan, build, test, and even communicate autonomously — are becoming part of the delivery team. Instead of being tools used by developers, agents are becoming collaborators with developers.

At the heart of this shift are large language models. They’re not just autocomplete engines anymore — they’re taking on structured tasks with contextual awareness, decision-making capabilities, and the ability to synthesize across data, code, and human intent.

What’s Changing for Agile Teams?
1. User Stories Become Living Entities
Product managers no longer need to manually decompose every idea. Agentic systems, powered by LLMs, can take high-level requirements and:

Break them into epics and user stories.
Prioritize based on past data and current goals.
Auto-generate acceptance criteria and wireframes.
Example: A prompt like “Build a real-time dashboard for store inventory” results in a backlog-ready story with suggested API endpoints, frontend component architecture, and testing scenarios — within minutes.

2. Autonomous Code Sprints
Code generation is no longer limited to isolated suggestions. Agents can:

Scaffold full services, frontends, and pipelines.
Collaborate through Git workflows.
Validate code against defined architecture and style guides.
A single agent might own the login service: building it, writing unit tests, running CI/CD pipelines, and raising a PR — all before the team standup.

3. Testing is Continuous, Agent-First
Test coverage gaps? Edge case regressions? Load simulations? Agents now:

Generate and run test suites based on requirements.
Identify likely failure points using past incidents.
Integrate with CI tools and raise alerts autonomously.
Testing becomes less reactive and more proactive — machine intuition reduces human oversight.
"""

result = chain.invoke({'text':text})
print(result)