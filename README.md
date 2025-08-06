# Pydantic Output Parser
1. Pydantic Output parser is a tool in LangChain that:
    -   Uses a Pydantic model to define the expected structure of the LLM's output.
    -   Ensures the output from LLM is parsed and validated into structured, type-safe Python objects.
    -   Automatically provides formatting instructions to LLM so that it knows how to return data in a machine readable way.

  **Points to Note**: Define structure of output, output is parsed, validated into structured, type-safe python object.

**Why Use It?**
Without a parser, LLMs return unstructured text that's hard to trust or automate. With PydanticOutputParser, you can:

- Strict Schema Enforcement --> Ensures that LLM responses follow a well-defined structure.
- Type Safety --> Automatically converts LLM outputs into Python objects.
- Easy Validation --> Uses Pydantic's built-in validation to catch incorrect or missing data.
- Seamless Integration --> Works well with other LangChain components.


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
1. llm.invoke(formatted_prompt) -> sends to LLM.
1. parser.parse(output) -> parses LLM result into Pydantic model.


**Advanced Usage**

You can chain multiple-steps like:

-   chain1 = prompt1|llm|parser1
-   chain2 = prompt2|llm|parser2
-   combined_chain = chain1|chain2

Or wrap it with RunnableSequence:

-   multi-step = RunnableSequence(first=chain1,then=chain2)


**Supported Components in chain**
|Component|Description|
|-------|--------------|
|PromptTemplate / ChatPromptTemplate|	Template engine|
|LLM (OpenAI, ChatOpenAI, etc.)|Language model|
|OutputParser (Pydantic, StrOutput, etc.)|Structured output|
|Tools|Functions or APIs the LLM can call|
|Memory|Keeps track of state|
|Functions|Custom Python logic (RunnableLambda)|

**Chain Architecture**


                          ┌────────────────────────────┐
                          │        User Input          │
                          │ ────────────────────────── │
                          │ contract_text (raw text)   │
                          └────────────┬───────────────┘
                                       │
                                       ▼
                      ┌────────────────────────────────────┐
                      │       ChatPromptTemplate           │
                      │ ─────────────────────────────────  │
                      │ "Extract clauses from: {text}..."  │
                      │ + {format_instructions}            │
                      └────────────┬───────────────────────┘
                                   │
                                   ▼
                        ┌────────────────────────────┐
                        │        ChatOpenAI          │
                        │ ────────────────────────── │
                        │   gpt-4 / gpt-3.5-turbo     │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                 ┌────────────────────────────────────────────┐
                 │     Raw LLM Output (Text in JSON Format)    │
                 └────────────┬────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │             PydanticOutputParser (OutputParser)        │
        │ ─────────────────────────────────────────────────────── │
        │ Parses the raw text into Python objects (ContractAnalysis) │
        └────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │     Final Structured Output (Python)   │
        │ ────────────────────────────────────── │
        │ clauses: List[Clause(title, content)] │
        └────────────────────────────────────────┘


**What's happening at each stage**

|Step|Component|Role|
|-------|--------------|------|
|1|Input|Raw contract text|
|2|ChatPromptTemplate|Formats prompt with role-base messages and get_format_instructions()|
|3|ChatOpenAI|Sends formatted prompt to GPT-4/GPT-3.5|
|4|Raw LLM Response|JSON string of clauses (as text)|
|5|PydanticOutputParser|Convert JSON String -> Python object|
|6|Final Output|ContractAnalysis object with list of structured clauses|


**Why this matters**
*   Prompt and LLM are decoupled
*   You can swap or reuse components
*   Parsing ensures structure and validation
*   Chains are modular and testable
*   You can combine chains for multi-step flows.


# Invoke()
The invoke() method is how you run a chain, model, prompt, or tool in LangChain

Syntax:
    
    .invoke(input: dict or stirng) -> output

It executes the pipeline and returns the final output - whether it's a string, JSON object, or a Python class.


**What can you call .invoke() on?**

|Object Type|.invoke() Works?|What it returns|
|-------|--------------|------|
|PromptTemplate|✅ Yes|Rendered string prompt|
|ChatPromptTemplate|✅ Yes|List of AIMessage/HumanMessage|
|ChatOpenAI|✅ Yes|LLM Output|
|PydanticOutputParser|✅ Yes|Parsed Pydantic Object|
|RunnableSequence(Chain)|✅ Yes|Final result from full chain|


**Example:**
    
    chain = prompt | llm | parser

    output = chain.invoke({
        "contract_text": "This Agreement is made on ...",
        "format_instructions":parser.get_format_instructions()
    })

**Behind the scenes:**
1. prompt.invoke(...) -> returns formatted prompt(with variables filled)
2. llm.invoke(prompt_output) -> sends to GPT-4 and get raw output
3. parser.invoke(llm_output) -> parses JSON into a python object

So chain.invoke(...) runs all steps in sequence and returns the final parsed output.


**Example:On Prompt Only**
    
    from langchain.prompts import PromptTemplate
    
    prompt = PromptTemplate.from_template("Hello, {name}")
    output = prompt.invoke({"name","Ravi"})
    print(output)


**Example: On ChatPromptTemplate**
    
    from langchain.prompt import ChatPromptTemplate
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "you are an assistant"),
        ("human","What is the capital of {country}")
    ])
    output = chat_prompt.invoke({"country":"France"})
    print(output)


**Common Use cases of .invoke()**

|Use Case|Input|Output|
|-------|--------------|------|
|Run prompt alone|dict of variables|Final prompt string or message|
|Run LLM directly|string or messages|LLM response text or message|
|Run full chain|dict of variables|Final Strcutred output|
|Run multi-step logic|Output of one step -> input of next|Automatically piped|


**Gotchas to Avoid**

|Mistake|Fix|
|-------|--------------|
|Missing input variables|Ensure all {var} placeholders are filled|
|Wrong input format|ChatPromptTemplate -> expects dict, not string|
|Using invoke() on unsupported objects|Only use on Runnable components|


**Bonus: How it compares to .stream() and .batch()**

|Method|Use When|
|-------|--------------|
|.invoke()|You want a single result|
|.batch()|You want to run multiple inputs|
|.stream()|You want token-by-token streaming|



# Output Parsers
Output parser in LangChain help convert raw LLM responses into structred format like JSON, CSV, Pydantic models, and more. They ensure consistency, validation and ease of user in application

**Why Output Parsers Matter**
By Default LLMs return plain text. Parsers help you:
-   Extract structured data
-   Enforce consistency
-   Make LLM outputs machine-usable
-   Enable chaining of components in complex workflows

**Common Output Parsers in LangChain**
-   String Output Parsers(StrOutputParser) --> Returns raw string output
-   Json Output Parsers 
-   Structured Output Parser --> Parses into a dictionary based on schema or example
-   Pydantic Output parser --> Parses output into a Pydantic model(typed, structured)
-   CommaSeparatedListOutputParser --> Returns list of strings split by commas
-   RetryWithErrorOutput
Parser --> Retries formatting if LLM output is invalid

**When to use Each Parser**

|Goal|Use This Parser|
|-------|--------------|
|Just need the text|StrOutputParser|
|Need validated structure|PydanticOutputParser|
|Need a list of items|CommaSeparatedListOutputParser|
|Using tools/agents|JsonOutputKeyToolsParser|
|Retry on bad formatting|RetryWithErrorOutputParser|


**StrOutputParser**

The StrOutputParser is the simplest output parser in LangChain. It is used to parse the output of the LLM and return it as a plain string.

**StructuredOutputParser**

StructuredOutputParser is an output parser in LangChain that helps extract structured JSON data from LLM responses based on predefined field schemas.
It works by defining a list of fields (ResponseSchema) that the model should return, ensuring the output follows a structured format.


**Draw Chain**

chain.get_graph().print_ascii()

    +-------------+       
     | PromptInput |       
     +-------------+       
            *              
            *              
            *              
    +----------------+     
    | PromptTemplate |     
    +----------------+     
            *              
            *              
            *              
      +------------+       
      | ChatOpenAI |       
      +------------+       
            *              
            *              
            *              
   +-----------------+     
   | StrOutputParser |     
   +-----------------+     
            *              
            *              
            *              
+-----------------------+  
| StrOutputParserOutput |  
+-----------------------+  


**Sequential Chain**

 Topic -> LLM -> Report -> LLM -> Summary

    +-------------+       
     | PromptInput |       
     +-------------+       
            *              
            *              
            *              
    +----------------+     
    | PromptTemplate |     
    +----------------+     
            *              
            *              
            *              
      +------------+       
      | ChatOpenAI |       
      +------------+       
            *              
            *              
            *              
   +-----------------+     
   | StrOutputParser |     
   +-----------------+     
            *              
            *              
            *              
+-----------------------+  
| StrOutputParserOutput |  
+-----------------------+  
            *              
            *              
            *              
    +----------------+     
    | PromptTemplate |     
    +----------------+     
            *              
            *              
            *              
      +------------+       
      | ChatOpenAI |       
      +------------+       
            *              
            *              
            *              
   +-----------------+     
   | StrOutputParser |     
   +-----------------+     
            *              
            *              
            *              
+-----------------------+  
| StrOutputParserOutput |  
+-----------------------+  


**Parallel Chain**

            +---------------------------+            
            | Parallel<notes,quiz>Input |            
            +---------------------------+            
                 **               **                 
              ***                   ***              
            **                         **            
+----------------+                +----------------+ 
| PromptTemplate |                | PromptTemplate | 
+----------------+                +----------------+ 
          *                               *          
          *                               *          
          *                               *          
  +------------+                    +------------+   
  | ChatOpenAI |                    | ChatOpenAI |   
  +------------+                    +------------+   
          *                               *          
          *                               *          
          *                               *          
+-----------------+              +-----------------+ 
| StrOutputParser |              | StrOutputParser | 
+-----------------+              +-----------------+ 
                 **               **                 
                   ***         ***                   
                      **     **                      
           +----------------------------+            
           | Parallel<notes,quiz>Output |            
           +----------------------------+            
                          *                          
                          *                          
                          *                          
                 +----------------+                  
                 | PromptTemplate |                  
                 +----------------+                  
                          *                          
                          *                          
                          *                          
                   +------------+                    
                   | ChatOpenAI |                    
                   +------------+                    
                          *                          
                          *                          
                          *                          
                +-----------------+                  
                | StrOutputParser |                  
                +-----------------+                  
                          *                          
                          *                          
                          *                          
              +-----------------------+              
              | StrOutputParserOutput |              
              +-----------------------+   


**Branch Chain**

   +-------------+      
    | PromptInput |      
    +-------------+      
            *            
            *            
            *            
   +----------------+    
   | PromptTemplate |    
   +----------------+    
            *            
            *            
            *            
     +------------+      
     | ChatOpenAI |      
     +------------+      
            *            
            *            
            *            
+----------------------+ 
| PydanticOutputParser | 
+----------------------+ 
            *            
            *            
            *            
       +--------+        
       | Branch |        
       +--------+        
            *            
            *            
            *            
    +--------------+     
    | BranchOutput |     
    +--------------+     


**Chain**

|Chain Name|Description|
|-------|--------------|
|LLMChain|Basic chain that calls an LLM with a prompt template|
|Sequential Chain|Chain multiple LLM calls in a specific sequence|
|SimpleSequentialChain|A simplified version of Sequential Chain for easier use|
|ConversationalRetrievalChain|Handles conversational Q & A with memory and retrieval|
|RetrievalQA|Fetches relevant documents and uses an LLM for question-answering|
|RouterChain|Directs user queries to different chains based on intent|
|MultiPromptChain|Uses different prompts for different user intents dynamically|
|HydeChain(Hypothetical Document Embedding)|Generates hypothetical answers to improve document retrieval|
|AgentExecutorChain|Orchestrates different tools and actions dynamically using an agent|
|SQLDatabaseChain|Connects to SQL databases and answers natural language queries|


**Problem**

Too many chains include 2 major problem:
-   Big codebase
-   increased learning curve

**Runnables**

Unit to work
-   input
-   process
-   output

Common Interface
-   invoke
-   batch
-   stream

Connect to make complex workflow

1. Task Specific Runnables

    These are core LangChain components that have been converted into Runnables so they can be used in pipeline.
    
    Purpose: Perform task-specific operations like LLM calls, prompting, retrieval.
    Examples: ChatOpenAI, PromptTemplate, Retriever

2. Runnable Primitives

    These are fundamental building blocks for structuring execution logic in AI workflows.
    Purpose: They help orchestrate execution by defining how differnt Runnables interact(sequentially, in parallel, conditionally, etc)

    Examples:
    -   RunnableSequence -> Runs steps in order(| operator)
    -   RunnableParallel -> Runs multiple steps simultaneously.
    -   RunnableMap -> Maps the same input across multiple functions.
    -   RunnableBranch -> Implements conditional execuition(if-else logic)
    -   RunnableLambda -> Wraps custom Python functions into Runnables.
    -   RunnablePassThrough -> Just forwards input as output(acts as a placeholder)


    1. Runnable Sequence
        Runnable Sequence is sequential chain of runnables in LangChain that executes each step one after another,
        passing the output of one step as the input to the next.
        It is useful when you need to compose multiple runnables together in a structured workflow.

    2.  RunnableParallel
        Runnable Parallel is runnable primitive that allows multiple runnables to execute in parallel.
        Each runnable receives the same input and processes it independently, producing a dictionary of outputs.

    3.  Runnable Passthrough
        Runnable Passthrough is a special Runnable primitive that simply returns the input as output without modifying it.

    4.  Runnable Lambda
        Runnable Lambda is a runnable primitive that allows you to apply custom Python functions within an AI pipeline

        It acts as a middleware between different AI components, enabling preprocessing, transformation, API calls, filtering and post-processing in a Langchain workflow.
         The RunnableLambda allows us to define custom functions that can be integrated into the LangChain workflow.
         This is useful for tasks like word counting, data validation, or any custom processing that needs to be done on the output of a chain.
         The RunnableLambda can be used anywhere in the chain, allowing for flexible and powerful workflows.
         The RunnableLambda can be used to create custom processing steps in a LangChain workflow.
         This allows for more complex and tailored workflows that can adapt to specific needs.
         The RunnableLambda can be used to create custom processing steps in a LangChain workflow.
         This allows for more complex and tailored workflows that can adapt to specific needs.
         The RunnableLambda can be used to create custom processing steps in a LangChain workflow.
         This allows for more complex and tailored workflows that can adapt to specific needs.      
    
    5.  RunnableBranch
        RunnableBranch is a control flow component in LangChain that allows you to conditionally route input data to different chains or runnables based upon custom logic.

        It functions like an if/elif/else block for chains - where you define a set of condition functions, each associated with a runnable(e.g LLM call, prompt chain, or tool). The first matching condition is executed. If no condition matches, a default runnable is used (if provided).
    
    