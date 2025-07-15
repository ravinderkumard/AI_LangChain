# Pydantic Output Parser
1. Pydantic Output parser is a tool in LangChain that:
    -   Uses a Pydantic model to define the expected structure of the LLM's output.
    -   Ensures the output from LLM is parsed and validated into structured, type-safe Python objects.
    -   Automatically provides formatting instructions to LLM so that it knows how to return data in a machine readable way.


**Why Use It?**
Without a parser, LLMs return unstructured text that's hard to trust or automate. With PydanticOutputParser, you can:


|Problem|Solved By|
|-------|--------------|
|LLMs returning inconsistent output|Enforced JSON schema via Pydantic|
|Parsing raw text manually|No need -- it auto-parse|
|Type-checking and validation|Built-in via Pydantic|
|Prompting the LLM to use correct format|get_format_instrcutions() helps guide the model|


**How it Works(Step by Step)**
Step1 : Define the expected output using Pydantic
    from pydantic import BaseModel, Field

    class Person(BaseModel):
        name: str = Field(description="The full name")
        age: int = Field(description="The person's age")

Step 2: Create a Pydantic Output Parser
    from langchain.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=Person)

Step 3: Get formatting instructions for LLM
    format_instructions = parser.get_format_instructions()

    This tells the LLM:
    "Please output the JSON using this exact schema..."

Step 4: Add it to a prompt and invoke the chain

    from langchain.prompts import ChatPromptTemplate
    from langchain.chat_models import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "Give me a person's name and age. {format_instructions}")
    ])

    llm = ChatOpenAI(model="gpt-4")
    chain = prompt | llm | parser

    response = chain.invoke({"format_instructions": format_instructions})
    print(response)

**Output**
Person(name="Alice Johnson", age=32)

**Benefits in Production**

|Benefit|Why it's matters|
|-------|--------------|
|Structure| You can trust the LLM's output schema|
|Validation|Automatically catches bad/missing fields|
|Type Safety|You get real Python objects, not messy strings|
|JSON Ready|Easy to store, send to APIs, or write to DBs|
|Scalable|Use it in chains, agents, pipelines|


# get_format_instructions()
get_format_instructions() is a method provided by LangChain's output parsers(especially PydanticOutputParser) that returns human-readable instructions for how the LLM should format its output.

**Why do you need it?**
LLMs like GPT are flexible, but often too flexible - they'll return answer in paragraphs, bullet points, tables, or other formats depending on how they're prompted.
In production use cases, we want structured JSON - consistently and reliably.
Use get_format_instructions():
    1. You extract the expected JSON schema in plain english.
    2. You include that in your prompt.
    3. The LLM now knows the exact format to follow.
    4. The PydanticOutputParser can parse the result safely into Python objects.


**Include in Prompt**
    from langchain.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an assistant."),
        ("human", "Describe a product.\n{format_instructions}")
    ])
    Now the LLM knows:
    -   What to output
    -   What fields are required
    -   What format is valid (JSON in a code block)


**Benefits of Using get_format_instructions()**
|Benefit|Why it's useful|
|-------|--------------|
|Reliable Output| Guide the LLM to follow a strict format|
|No Guesswork| You don't have to manually write the format spec|
|Easier Parsing| Ensure output matches what your parser expects|
|Schema-Aware Prompts| Makes LLMs behave like structured data emitters|
|Supports Automation|Ideal for APIs, backend services and UI rendering|


**Without it**
The product is a Laptop that costs $1200.


**With it**
{
  "name": "Laptop",
  "price": 1200.0
}



# 💬 PromptTemplate 
**Use When**
-   You're using non-chat/completion-based models.
-   You want a single formatted string
-   You don't need roles like "system" or "user"

Example:
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        "Summarize this article:\n\n{article_text}"
    )

    formatted_prompt = prompt.format(article_text="The economy is growing...")
**Good For:**
-   Zero-shot summarization
-   Embedding model inputs.
-   Traditional NLP prompts
-   Use with models like text-davinci-003, GPT4ALL, etc.

# 💬 ChatPromptTemplate 
**Use When**
-   You're using chat based models(gpt-3.5-turbo, gpt-4, Claude,etc)
-   You want to structure input as a conversation (system/human/assistant messages)
-   You need role separation, memory, or multi-turn interactions.

Example:
    from langchain.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "Tell me a fun fact about {topic}.")
    ])

    formatted_prompt = prompt.format_messages(topic="space")

**Good For:**
-   Complex multi-step agents
-   Role-based control(system instruction)
-   Chat memory/context
-   Fine-grained control with OpenAI Chat API



|Mistake|What Happens|
|-------|--------------|
|Use PromptTemplate with ChatOpenAI|May work, but context formatting will be ignored|
|Use ChatPromptTemplate with regular LLM (OpenAI)|Will raise an error — only accepts plain strings|


|Model Type|Use|
|-------|--------------|
|ChatOpenAI, ChatAnthropic, etc.|	✅ ChatPromptTemplate|
|OpenAI (text-davinci-003), GPT4All, etc.|✅ PromptTemplate|





# What is Chain in LangChain?
A chain in LangChain is a sequence of components that process data in a specific order. Each component can be a prompt, model, or output parser, and they work together to transform input into structured output. Chains allow for modular and reusable workflows, enabling complex tasks to be broken down into simpler steps. In this case, the chain combines a prompt for invoice extraction, an LLM for processing, and an output parser to structure the results.

A pipeline that connects mutliple components like :
   - prompts
   - models
   - Output Parser
   - tools
   - functions
The output of one component can be the input to another, creating a flow of data through the chain.

**Basic Chain Flow**
    Input -> PromptTemplate -> LLM -> OutputParser -> Final Output

    chain = prompt | llm | parser

    This creates a Runnable Sequence.


    -   Accepts {text} as input
    -   Builds a structured prompt with instructions.
    -   Sends it to GPT-4
    -   Parses and validates the result.

**What Happens Internally**
LangChain converts this:
    chain = prompt | llm | parser

Into this internal execution:
    1. prompt.format(input) -> fills the {contract_text} and {format_instructions}
    2. llm.invoke(formatted_prompt) -> sends to LLM.
    3. parser.parse(output) -> parses LLM result into Pydantic model.


**Advanced Usage**
You can chain multiple-steps like:
    chain1 = prompt1|llm|parser1
    chain2 = prompt2|llm|parser2
    combined_chain = chain1|chain2

Or wrap it with RunnableSequence:
    multi-step = RunnableSequence(first=chain1,then=chain2)

**Supported Components in chain**
|Component|Description|
|-------|--------------|
|PromptTemplate / ChatPromptTemplate|	Template engine|
|LLM (OpenAI, ChatOpenAI, etc.)|Language model|
|OutputParser (Pydantic, StrOutput, etc.)|Structured output|
|Tools|Functions or APIs the LLM can call|
|Memory|Keeps track of state|
|Functions|Custom Python logic (RunnableLambda)|
