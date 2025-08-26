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
    

    6.  LCEL
        LangChain expression Language
        
            RunnableSequence -> r1|r2|....
        


# RAG
RAG is a technique that combines information retrieval with language generation, where a model retrieve relevant documents from a knowledge base and then uses them as context to generate accurate and grounded responses.
-   Document Loaders
-   Text Splitters
-   Vector Databases
-   Retrievers

## Document Loaders
Document loaders are components in LangChain used to load data from various sources into a standardized format(usually as Document objects), which can then be used for chunking, embedding, retrieval and generation.

    Document(
        page_content="text content",
        metadata={"source":"filename.pdf"...}
    )

-   TextLoader
    TextLoader is a simple and commonly used document loader in langChain that reads plain text files and converts them into LangChain Document objects.

    Use Case
    *   Ideal for loading chat logs, scraped text, transcripts, code snippets, or any plain text data into a LangChain pipeline.

    Limitation
    *   Works only with .txt files

-   PyPDFLoader
    PyPDFLoader is a document loader in LangChain used to load content from PDF files and convert each page into a Document object.

    [
        Document(page_content="Text from page 1", metadata={"page":0,"source":"file.pdf"}),
        Document(page_content="Text from page 2", metadata={"page":1,"source":"file.pdf"}),
    ]

    Limitations:
    *   It uses the PyPDF library under the hood - not greate with scanned PDFs or complex layouts.

    Use Cases:
    https://python.langchain.com/docs/integrations/document_loaders/#pdfs

-   DirectoryLoader
    DirectoryLoader is a document loader in LangChain that recursively loads all files from a specified directory and its subdirectories, converting each file into a Document object.

    Use Cases:
    *   Ideal for loading large datasets, collections of documents, or entire directories of text files into a LangChain pipeline.
    *   Automatically handles multiple file types like .txt, .pdf, .csv, etc.

    Limitations:
    *   Requires specifying the file extensions to load.
    *   May not handle complex directory structures well.

    ### Load vs Lazy Load
-   Load: Loads all documents immediately into memory.
    *   Useful for small datasets where you want quick access to all documents.
    *   Can be more memory-intensive as it loads everything at once.
    *   Eager Loading: Loads all documents at once, which is faster for small datasets but can consume more memory.

-   Lazy Load: Loads documents on-demand, only when accessed, which is more memory efficient for large datasets.
    *   Useful for large datasets where you want to avoid loading everything into memory at once.
    *   Can be slower for initial access since it loads documents as needed.
    *   Lazy Loading: Loads documents only when accessed, which is more memory efficient for large datasets but can be slower for initial access.
    *   Returns: A generator of document objects, allowing you to iterate through documents without loading them all into memory at once.



-   WebBaseLoader
    WebBaseLoader is a document loader in LangChain that loads web pages and extracts their content into Document objects. It supports loading multiple pages from a list of URLs or a sitemap.

    Limitations:
    *   Requires internet access to fetch web pages.
    *   May not handle complex web pages with heavy JavaScript or dynamic content.
    Use Cases:
    *   Ideal for scraping content from multiple web pages, blogs, or news sites.
    *   Can be used to create datasets from online articles, documentation, or any publicly accessible web content.

        from langchain.document_loaders import WebBaseLoader

        urls = ["https://example.com/page1", "https://example.com/page2"]
        loader = WebBaseLoader(urls=urls)
        documents = loader.load()

        ### Output: List of Document objects with page content and metadata
        [
            Document(page_content="Content of page 1", metadata={"source":"https://example.com/page1"}),
            Document(page_content="Content of page 2", metadata={"source":"https://example.com/page2"}),
        ]
-   CSVLoader
    CSVLoader is a document loader in LangChain that reads CSV files and converts each row into a Document object. It supports loading data from local CSV files or remote URLs.

    Use Cases:
    *   Ideal for loading structured data from CSV files, such as tabular datasets, spreadsheets, or any comma-separated values.
    *   Can be used to create datasets for analysis, training, or retrieval tasks.

    Limitations:
    *   Works only with .csv files.
    *   May not handle complex CSV formats with nested structures or special characters well.

        from langchain.document_loaders import CSVLoader

        loader = CSVLoader(file_path="data.csv")
        documents = loader.load()

        ### Output: List of Document objects with row content and metadata
        [
            Document(page_content="Row 1 content", metadata={"source":"data.csv"}),
            Document(page_content="Row 2 content", metadata={"source":"data.csv"}),
        ]


## Text Splitters
Text splitters are components in LangChain that take raw text and break it into smaller, manageable chunks or segments. This is useful for processing large documents, improving retrieval accuracy, and ensuring LLMs can handle input effectively.
-   RecursiveCharacterTextSplitter
    RecursiveCharacterTextSplitter is a text splitter in LangChain that recursively splits text into smaller chunks based on character length, while preserving logical boundaries like sentences or paragraphs.

    Use Cases:
    *   Ideal for processing large documents, articles, or books where you want to maintain context within chunks.
    *   Helps improve retrieval accuracy by ensuring chunks are coherent and meaningful.

    Limitations:
    *   May not handle very large documents efficiently due to recursive splitting.
    *   Requires careful configuration of chunk size and overlap parameters.

    **Overcoming Model Limitations**
    Many embedding models and LLMs have token limits (e.g., 4096 tokens for GPT-3.5). Text splitters help ensure that input text is within these limits by breaking it into smaller, manageable chunks.
    **Downstream Tasks**
    Text splitters are often used before tasks like:
    -   Document retrieval
    -   Question answering
    -   Summarization
    -   Text generation
    
    Tasks.             Why Text Splitters?
|Task|Reason|
|-------|--------------|
|Document Retrieval|Ensures chunks are small enough for efficient search|
|Question Answering|Maintains context within chunks for accurate answers|
|Summarization|Breaks long texts into manageable segments for summarization|
|Text Generation|Prepares input text for LLMs by splitting into coherent segments|
|Semantic Search|Improves search accuracy by chunking text into meaningful segments|
|Embedding Generation|Ensures input text is within token limits for embedding models|


**Optimizing Computational Resources**
Working with smaller chunks of text can be more memory-efficient and faster for processing, especially when dealing with large datasets or documents. Text splitters help optimize resource usage by breaking text into smaller, manageable pieces.


### Text Splitters
-   Length Based Text Splitter
    LengthBasedTextSplitter is a text splitter in LangChain that splits text into chunks based on a fixed character length. It ensures that each chunk does not exceed the specified length, making it suitable for tasks where input size needs to be controlled.

    Use Cases:
    *   Ideal for processing large documents where you want to limit chunk size for LLMs or embedding models.
    *   Helps maintain consistency in input size across different chunks.

    Limitations:
    *   May not preserve logical boundaries like sentences or paragraphs.
    *   Requires careful configuration of chunk length to avoid cutting off important context.

    - Chunk Overlap
    Chunk overlap is a technique used in text splitting where a portion of the text is repeated in adjacent chunks. This helps maintain context and continuity between chunks, especially when processing large documents or texts with complex structures.

    Use Cases:
    *   Ideal for tasks like question answering or summarization where context is important.
    *   Helps improve retrieval accuracy by ensuring that related information is not lost between chunks.
    Limitations:
    *   Increases the size of the dataset as overlapping text is repeated.
    *   Requires careful configuration of overlap size to balance context preservation and chunk size.

-   Text-Structured Based
    RecursiveCharacterTextSplitter is a text splitter in LangChain that recursively splits text into smaller chunks based on character length, while preserving logical boundaries like sentences or paragraphs.
    Use Cases:
    *   Ideal for processing large documents, articles, or books where you want to maintain context within chunks.
    *   Helps improve retrieval accuracy by ensuring chunks are coherent and meaningful.
    Limitations:
    *   May not handle very large documents efficiently due to recursive splitting.
    *   Requires careful configuration of chunk size and overlap parameters.
    
-   Document-Structured Based
    Document-structured text splitters are designed to split text based on the structure of the document, such as paragraphs, sections, or chapters. They ensure that chunks maintain logical boundaries and context, making them suitable for tasks like document retrieval, summarization, and question answering.

    Use Cases:
    *   Ideal for processing structured documents like reports, articles, or books.
    *   Helps maintain context and coherence within chunks for better retrieval and understanding.

    Limitations:
    *   May not handle unstructured text well.
    *   Requires knowledge of the document structure to configure splitting effectively.

-   Semantic Text Splitters
    Semantic text splitters are designed to split text based on semantic meaning rather than just character length or structure. They use natural language processing techniques to identify logical boundaries, such as sentences or paragraphs, while preserving the overall meaning of the text.

    Use Cases:
    *   Ideal for processing complex texts where maintaining semantic context is crucial.
    *   Helps improve retrieval accuracy by ensuring chunks are coherent and meaningful.

    Limitations:
    *   May require more computational resources for semantic analysis.
    *   Can be slower than traditional text splitters due to the complexity of semantic processing.


## Vector Stores
Vector stores are specialized databases designed to efficiently store and retrieve high-dimensional vectors, which are often used in machine learning and natural language processing tasks. They enable fast similarity search, nearest neighbor search, and other operations on vector data.

### Key Features of Vector Stores
-   High-dimensional vector storage: Vector stores can handle vectors with thousands of dimensions, making them suitable for complex data representations like word embeddings or image features. in-memory or on-disk storage.
-   Efficient indexing: They use advanced indexing techniques like tree-based structures, hash-based methods, or graph-based approaches to optimize search performance.
-   Similarity search: Vector stores provide efficient algorithms for finding similar vectors based on distance metrics like cosine similarity, Euclidean distance, or dot product.
-   Scalability: They can handle large datasets with millions or billions of vectors, making them suitable for big data applications.
-   Integration with machine learning frameworks: Vector stores can be easily integrated with popular machine learning libraries like TensorFlow, PyTorch, or scikit-learn for training and inference tasks.

    Key Features of Vector Stores
-   High-dimensional vector storage: They can handle vectors with thousands of dimensions, making them suitable for complex data representations like word embeddings or image features.
-   In-memory or on-disk storage: Vector stores can be configured to store vectors in memory for fast access or on disk for persistence.

### Use Cases

|Use Case|Description|
|-------|--------------|
|Similarity Search|Finding similar items based on vector representations, such as finding similar images or documents.|
|Recommendation Systems|Recommending items based on user preferences and item embeddings, such as movie or product recommendations.|
|Natural Language Processing|Storing and retrieving word embeddings, sentence embeddings, or document embeddings for tasks like text classification, sentiment analysis, or question answering.|    
|Image Retrieval|Storing and retrieving image features for tasks like image search, object detection, or image classification.|
|Anomaly Detection|Detecting anomalies in high-dimensional data by comparing vector representations, such as fraud detection or network intrusion detection.|

## Vector Store Types
|Vector Store|Description|
|-------|--------------|
|FAISS|Facebook AI Similarity Search - a popular open-source library for efficient similarity search and clustering of dense vectors.|
|Pinecone|A managed vector database service that provides fast and scalable vector search capabilities with built-in indexing and querying features.|
|Weaviate|An open-source vector database that combines vector search with semantic search capabilities, allowing for efficient retrieval of similar items based on their vector representations.|
|Milvus|An open-source vector database designed for high-performance similarity search and analytics on large-scale vector data.|
|Chroma|An open-source vector database that provides efficient storage and retrieval of high-dimensional vectors, optimized for machine learning applications.|
|Redis|A popular in-memory data store that supports vector search capabilities through its RedisAI module, allowing for fast similarity search and retrieval of high-dimensional vectors.|
|Elasticsearch|A distributed search and analytics engine that supports vector search capabilities through its k-NN plugin, allowing for efficient retrieval of similar items based on vector representations.|
|Qdrant|An open-source vector database that provides efficient storage and retrieval of high-dimensional vectors, optimized for real-time applications and machine learning tasks.|
|Annoy|A C++ library with Python bindings for approximate nearest neighbor search in high-dimensional spaces, designed for fast similarity search and retrieval of dense vectors.|
|ScaNN|A Google-developed library for efficient similarity search and clustering of high-dimensional vectors, optimized for large-scale datasets and machine learning applications.|        

## Vector Store Indexing
Vector store indexing is the process of organizing and optimizing high-dimensional vectors in a vector database to enable efficient similarity search, retrieval, and analytics. It involves creating data structures that allow for fast access to vectors based on their spatial relationships and distance metrics.
### Key Concepts
-   Vector Representation: Each item is represented as a high-dimensional vector, capturing its features or characteristics.
-   Distance Metrics: Measures like cosine similarity, Euclidean distance, or dot product are used to quantify the similarity between vectors.
-   Indexing Structures: Data structures like trees, graphs, or hash tables are used to organize vectors for efficient search and retrieval.
### Indexing Techniques
|Indexing Technique|Description|
|-------|--------------|
|Tree-based Indexing|Uses tree structures like KD-trees, Ball trees, or R-trees to partition the vector space and enable efficient search.|
|Hash-based Indexing|Uses hash functions to map vectors to buckets, allowing for fast retrieval of similar vectors based on hash collisions.|           
|Graph-based Indexing|Uses graph structures to represent relationships between vectors, enabling efficient traversal and search.|
|Product Quantization|Compresses high-dimensional vectors into lower-dimensional representations, reducing storage requirements and improving search speed.|
|Inverted Indexing|Creates an index that maps terms or features to vectors, enabling fast retrieval based on specific attributes or keywords.|
### Indexing Process
1. Vector Extraction: Convert items into high-dimensional vectors using techniques like embeddings, feature extraction, or manual encoding.
2. Distance Metric Selection: Choose an appropriate distance metric based on the nature of the data and the similarity search requirements.
3. Index Structure Creation: Build the index structure using one or more indexing techniques, optimizing for search speed and accuracy.
4. Index Population: Populate the index with vectors, ensuring efficient storage and retrieval.
5. Querying: Use the index to perform similarity search, retrieving vectors that are closest to a given query vector based on the selected distance metric.

## Vector Store vs Vector Database

-   Vector Stores
    *   Typically refers to lightweight library or service that focuses on storing vectors and performing similarity search.
    *   May not included many traditional database features like transactions, complex queries, or data management, role bases access control.
    *   Ideal for prototypes, small-scale applications, or specific use cases where high-dimensional vector storage and retrieval are the primary requirements.
    *   Examples: FAISS, Annoy, ScaNN, Chroma.
-   Vector Databases
    *   Refers to full-fledged database systems optimized for vector data, providing advanced features like indexing, querying, analytics, and data management.
    *   Includes capabilities like distributed architectures, scalability, role-based access control, and integration with various data sources.
    *   Suitable for complex applications requiring efficient storage, retrieval, and analysis of large-scale vector datasets.
    *   Examples: Pinecone, Weaviate, Milvus, Qdrant.
### Key Differences
    -   Purpose: Vector stores focus on storing and retrieving high-dimensional vectors, while vector databases provide a comprehensive database system optimized for vector data.
    -   Functionality: Vector stores primarily focus on similarity search and nearest neighbor search, while vector databases include data management, indexing, querying, and analytics capabilities.

|Aspect|Vector Store|Vector Database|
|-------|--------------|--------------|
|Purpose|Primarily for storing and retrieving high-dimensional vectors|Full-fledged database system optimized for vector data|
|Functionality|Focuses on similarity search and nearest neighbor search|Includes data management, indexing, querying, and analytics|
|Scalability|Can handle large datasets with millions or billions of vectors|Designed for high scalability, supporting distributed architectures|
|Integration|Often integrated with machine learning frameworks for training and inference|Can integrate with various data sources, APIs, and applications|
|Use Cases|Ideal for tasks like recommendation systems, image retrieval, and NLP|Suitable for complex applications requiring data management, analytics, and real-time querying|

### Vector Stores in LangChain
LangChain provides built-in support for several popular vector stores, allowing you to easily integrate them into your AI workflows. These vector stores can be used for tasks like document retrieval, similarity search, and embedding generation.

**Supported Stores**: LangChain integrates with multiple vector stores(FAISS, Pinecone, Weaviate, Milvus, Chroma, Qdrant, Redis, Elasticsearch) giving you flexibility in scale, features, and deployment.
**Common Interface**: A uniform Vector store API lets you swap out one backend for another with minimal code changes.
**Metadata Handling**: Most vector stores in langchain allow you to attach metadata to each document, enabling filter-based retrieval.

### Chroma Vector Store
Chroma is an open-source vector database designed for high-performance similarity search and analytics on large-scale vector data. It provides efficient storage and retrieval of high-dimensional vectors, optimized for machine learning applications.
### Key Features
-   High-dimensional vector storage: Chroma can handle vectors with thousands of dimensions, making it suitable for complex data representations like word embeddings or image features.
-   In-memory or on-disk storage: Chroma can be configured to store vectors in memory for fast access or on disk for persistence.
-   Efficient indexing: It uses advanced indexing techniques like tree-based structures, hash-based methods, or graph-based approaches to optimize search performance.
-   Similarity search: Chroma provides efficient algorithms for finding similar vectors based on distance metrics like cosine similarity, Euclidean distance, or dot product.
-   Scalability: Chroma can handle large datasets with millions or billions of vectors, making it suitable for big data applications.



### Retrievers
Retrievers are components in LangChain that facilitate the retrieval of relevant documents or information from a knowledge base or vector store based on a user's query. They play a crucial role in retrieval-augmented generation (RAG) systems, where the goal is to provide accurate and contextually relevant responses by leveraging external knowledge sources.
### Key Features of Retrievers
-   Query Processing: Retrievers process user queries to extract relevant keywords or phrases that can be used for searching the knowledge base.
-   Document Retrieval: They use various algorithms and techniques to search the knowledge base or vector store and retrieve documents that are most relevant to the query.
-   Ranking: Retrieved documents are often ranked based on their relevance to the query, using metrics like cosine similarity, Euclidean distance, or other distance measures.
-   Integration with LLMs: Retrievers can be integrated with large language models (LLMs) to provide context for generating responses, ensuring that the output is grounded in relevant information.
### Types of Retrievers
Categories of retrievers based on their retrieval methods:
    1. Vector-based Retrievers
    2. Keyword-based Retrievers
    3. Hybrid Retrievers

Categories of retrievers based on their Data Source:
    1. Local File-based Retrievers
    2. Database-based Retrievers
    3. API-based Retrievers
    4. Web-based Retrievers
    5. Wiki-based Retrievers

|Retriever Type|Description|
|-------|--------------|
|Vector-based Retrievers|Use vector representations of documents and queries to perform similarity search in a vector store.|
|Keyword-based Retrievers|Use keyword matching techniques to search for documents containing specific terms or phrases.|
|Hybrid Retrievers|Combine vector-based and keyword-based approaches to leverage the strengths of both methods.|
### Use Cases
|Use Case|Description|
|-------|--------------|
|Question Answering|Retrieving relevant documents to provide accurate answers to user queries.|
|Document Search|Finding documents related to a specific topic or keyword.|     
|Contextual Generation|Providing context for LLMs to generate more accurate and relevant responses.|
|Knowledge Base Augmentation|Enhancing the knowledge base with relevant information for improved retrieval and generation

### Wikipedia Retriever
The Wikipedia Retriever is a component in LangChain that allows you to retrieve relevant articles from Wikipedia based on a user's query. It leverages Wikipedia's vast knowledge base to provide accurate and contextually relevant information.
### Key Features
-   Query Processing: The Wikipedia Retriever processes user queries to extract relevant keywords or phrases for searching Wikipedia.
-   Article Retrieval: It uses Wikipedia's search API to find articles that are most relevant to the query.
-   Ranking: Retrieved articles are ranked based on their relevance to the query, using metrics like page views, relevance scores, or other factors.
-   Integration with LLMs: The Wikipedia Retriever can be integrated with large language models (LLMs) to provide context for generating    
responses, ensuring that the output is grounded in relevant information from Wikipedia.
### Use Cases
|Use Case|Description|
|-------|--------------|
|Question Answering|Retrieving relevant Wikipedia articles to provide accurate answers to user queries.|
|Document Search|Finding Wikipedia articles related to a specific topic or keyword.|     
|Contextual Generation|Providing context from Wikipedia for LLMs to generate more accurate and relevant responses.|
|Knowledge Base Augmentation|Enhancing the knowledge base with relevant information from Wikipedia for improved retrieval and generation.|      


### Vector Store Retriever
The Vector Store Retriever is a component in LangChain that allows you to retrieve relevant documents from a vector store based on a user's query. It leverages vector representations of documents and queries to perform efficient similarity search.
### Key Features
-   Query Processing: The Vector Store Retriever processes user queries to extract relevant keywords or phrases and convert them into vector representations using embedding models.
-   Document Retrieval: It uses similarity search algorithms to find documents in the vector store that are most similar to the query vector.
-   Ranking: Retrieved documents are ranked based on their similarity to the query, using distance metrics like cosine similarity, Euclidean distance, or dot product.
-   Integration with LLMs: The Vector Store Retriever can be integrated with large language models (LLMs) to provide context for generating responses, ensuring that the output is grounded in relevant information from the vector store.
### Use Cases
|Use Case|Description|
|-------|--------------|
|Question Answering|Retrieving relevant documents from the vector store to provide accurate answers to user queries.|
|Document Search|Finding documents in the vector store related to a specific topic or keyword.|     
|Contextual Generation|Providing context from the vector store for LLMs to generate more accurate and relevant responses.|
|Knowledge Base Augmentation|Enhancing the knowledge base with relevant information from the vector store for improved retrieval and generation.|   

### Maximal Marginal Relevance (MMR) Retriever
The Maximal Marginal Relevance (MMR) Retriever is a component in LangChain that implements the MMR algorithm to retrieve a diverse set of relevant documents based on a user's query. The MMR algorithm balances relevance and diversity, ensuring that the retrieved documents are not only relevant to the query but also diverse in content.
### Key Features
-   Query Processing: The MMR Retriever processes user queries to extract relevant keywords or phrases and convert them into vector representations using embedding models.
-   Document Retrieval: It uses similarity search algorithms to find an initial set of relevant documents from the vector store.
-   MMR Algorithm: The MMR algorithm is applied to the initial set of documents to select a subset that maximizes relevance to the query while minimizing redundancy among the selected documents.
-   Ranking: The final set of documents is ranked based on their relevance to the query and their diversity.
-   Integration with LLMs: The MMR Retriever can be integrated with large language models (LLMs) to provide context for generating responses, ensuring that the output is grounded in a diverse set of relevant information.


### Multi query Retriever
The Multi-Query Retriever is a component in LangChain that allows you to retrieve relevant documents from a vector store using multiple queries. This approach can help improve retrieval accuracy by leveraging different perspectives or aspects of the user's information need.
### Key Features
-   Multiple Query Processing: The Multi-Query Retriever processes multiple user queries, extracting relevant keywords or phrases from each query and converting them into vector representations using embedding models.
-   Document Retrieval: It uses similarity search algorithms to find documents in the vector store that are most similar to each query vector.
-   Aggregation: Retrieved documents from all queries are aggregated, and duplicates are removed to create a unified set of relevant documents.
-   Ranking: The aggregated documents are ranked based on their relevance to the queries, using distance metrics like cosine similarity, Euclidean distance, or dot product.
-   Integration with LLMs: The Multi-Query Retriever can be integrated with large language models (LLMs) to provide context for generating responses, ensuring that the output is grounded in relevant information from the vector store.


### Contextual Compression Retriever
The Contextual Compression Retriever is a component in LangChain that combines a primary retriever with a secondary retriever to enhance the retrieval of relevant documents. The primary retriever is used to fetch an initial set of documents, while the secondary retriever is used to refine and compress the context of these documents, ensuring that the most relevant information is retained.
### Key Features
-   Primary Retriever: The primary retriever fetches an initial set of relevant documents from the knowledge base or vector store based on the user's query.
-   Secondary Retriever: The secondary retriever processes the documents retrieved by the primary retriever, extracting and compressing the most relevant context.
-   Contextual Compression: The secondary retriever uses techniques like summarization, keyword extraction, or embedding-based filtering to reduce the amount of information while retaining the most important content.
-   Ranking: The compressed documents are ranked based on their relevance to the original query.
-   Integration with LLMs: The Contextual Compression Retriever can be integrated with large language models (LLMs) to provide context for generating responses, ensuring that the output is grounded in the most relevant and concise information. 


### More Retrievers
-   Time Weighted Retriever
-   SVM Retriever
-   BM25 Retriever
-   TF-IDF Retriever
-   Parent Document Retriever
-   Self Query Retriever
-   Multi Vector Store Retriever


## RAG
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation, where a model retrieves relevant documents from a knowledge base and then uses them as context to generate accurate and grounded responses. RAG enhances the capabilities of language models by providing them with access to external knowledge, allowing them to generate more informed and contextually relevant outputs.

**Problem with LLMs**: Large Language Models (LLMs) like GPT-3.5 or GPT-4 are powerful but have limitations in terms of knowledge cutoff dates and the ability to access up-to-date information. They may also generate plausible-sounding but incorrect or nonsensical answers.

**RAG Solution**: RAG addresses these limitations by integrating a retrieval component that fetches relevant documents from a knowledge base (like a vector store or database) based on the user's query. The retrieved documents are then used as additional context for the language model to generate responses.

**RAG Workflow**
1. User Query: The user submits a query or question.
2. Document Retrieval: A retriever component searches the knowledge base and retrieves relevant documents based on the query.
3. Contextual Generation: The retrieved documents are provided as context to the language model, which generates a response that is informed by the content of the documents.
4. Response Delivery: The generated response is returned to the user.   

**Fine-tuning vs RAG**
|Aspect|Fine-tuning|RAG|
|-------|--------------|--------------|
|Knowledge Update|Requires retraining the model with new data to update knowledge|Can access up-to-date information from the knowledge base without retraining|
|Contextual Understanding|Limited to the knowledge encoded during training|Can leverage external documents to provide contextually relevant responses|
|Flexibility|Less flexible, as it relies on the model's internal knowledge|More flexible, as it can adapt to    different knowledge bases or domains|   
|Complexity|Simpler, as it involves training a single model|More complex, as it involves integrating retrieval and generation components|   

**Supervised vs Unsupervised RAG**
|Aspect|Supervised RAG|Unsupervised RAG|
|-------|--------------|--------------|
|Training Data|Requires labeled data with query-document pairs for training|Does not require labeled data, relies on unsupervised learning techniques|
|Model Training|Involves training both the retriever and generator components|Focuses on training the retriever component, while the generator may be pre-trained|
|Performance|Can achieve higher accuracy with quality labeled data|May have lower accuracy due to lack of supervision, but can still be effective|
|Use Cases|Suitable for applications with available labeled
|Use Cases|Suitable for applications with available labeled data, such as customer support or FAQ systems|Suitable for applications where labeled data is scarce, such as open-domain question answering or knowledge discovery|

**RLHF
**: Reinforcement Learning from Human Feedback (RLHF) is a technique used to fine-tune language models based on feedback from human evaluators. In the context of RAG, RLHF can be used to improve the quality of generated responses by incorporating human preferences and judgments into the training process.

**Supervised Fine tuning Process**
1. Collect data: Gather a dataset of query-document pairs, where each query is associated with relevant documents from the knowledge base.
2. Choose a method: Select a supervised learning method, such as contrastive learning or cross-encoder training, to train the retriever component.
3. Train for a few epochs: Train the retriever model for a few epochs on the labeled dataset, optimizing for retrieval accuracy.
4. Evaluate and iterate: Evaluate the performance of the retriever on a validation set, and iterate on the training process as needed to improve results.

**Unsupervised Fine tuning Process**
1. Pre-train the retriever: Use unsupervised learning techniques, such as self-supervised learning or autoencoding, to pre-train the retriever component on a large corpus of unlabeled text.
2. Use pseudo-labels: Generate pseudo-labels by using the pre-trained retriever to retrieve documents for a set of queries, and use these pseudo-labels to fine-tune the retriever.
3. Train for a few epochs: Fine-tune the retriever model for a few epochs on the pseudo-labeled dataset, optimizing for retrieval accuracy.
4. Evaluate and iterate: Evaluate the performance of the retriever on a validation set, and iterate on the training process as needed to improve results.

## Problem with fine-tuning
-   Data Requirements: Fine-tuning requires a substantial amount of labeled data, which may not always be available or easy to obtain.
-   Computational Resources: Fine-tuning large language models can be computationally expensive and time-consuming, requiring significant hardware resources.
-   Overfitting: There is a risk of overfitting the model to the fine-tuning dataset, which can lead to poor generalization to new queries or domains.
-   Knowledge Cutoff: Fine-tuning does not address the knowledge cutoff issue, as the model's knowledge is still limited to the data it was trained on.
-   Maintenance: Keeping the model up-to-date with new information requires periodic re-fine-tuning, which can be resource-intensive.

### In-context Learning with RAG
In-context learning with RAG involves providing the language model with relevant context from retrieved documents to guide the generation of responses. This approach allows the model to leverage external knowledge without the need for fine-tuning. The retrieved documents serve as additional input, helping the model generate more accurate and contextually relevant answers.

### Emergent Properties with RAG
Emergent properties refer to the unexpected behaviors or capabilities that arise when combining retrieval and generation in RAG systems. For example, a RAG system may exhibit improved reasoning abilities, better handling of ambiguous queries, or the ability to synthesize information from multiple sources. These emergent properties can enhance the overall performance and utility of the RAG system.

### RAG = Information Retrieval + Text Generation
RAG combines the strengths of information retrieval and text generation to create a powerful system for answering queries. By retrieving relevant documents from a knowledge base and using them as context for a language model, RAG enables the generation of accurate, informed, and contextually relevant responses.

### Steps in RAG
1. Indexing: Create a knowledge base or vector store containing documents or information relevant to the domain of interest.
2. Retrieval: Use a retriever component to search the knowledge base and retrieve documents relevant to the user's query.
3. Augmentation: Provide the retrieved documents as context to the language model.
4. Generation: Use the language model to generate a response based on the query and the provided context.
5. Delivery: Return the generated response to the user.


### Indexing for RAG
Indexing for RAG involves creating a structured representation of the knowledge base or vector store to enable efficient retrieval of relevant documents. This process typically includes the following steps:
1. Document Collection: Gather a corpus of documents or information relevant to the domain of interest.
    Tools: LangChain Loaders
2. Text Preprocessing: Clean and preprocess the text data, including tokenization, normalization, and removal of stop words.
    Tools: LangChain Text Splitters
3. Embedding Generation: Convert the preprocessed text into high-dimensional vector representations using embedding models.
    Tools: LangChain Embeddings
4. Index Creation: Create an index structure to organize the vectors for efficient similarity search and retrieval.
    Tools: LangChain Vector Stores
5. Storage: Store the indexed vectors in a vector store or database for later retrieval.    
    Tools: LangChain Vector Stores

### Retrieval for RAG
Retrieval for RAG involves using a retriever component to search the indexed knowledge base or vector store and retrieve documents that are relevant to the user's query. This process typically includes the following steps:
1. Query Processing: Process the user's query to extract relevant keywords or phrases and convert them into a vector representation using an embedding model.
    Tools: LangChain Embeddings
2. Similarity Search: Use similarity search algorithms to find documents in the vector store that are most similar to the query vector.
    Tools: LangChain Retrievers
3. Ranking: Rank the retrieved documents based on their relevance to the query, using distance metrics like cosine similarity, Euclidean distance, or dot product.
    Tools: LangChain Retrievers
4. Selection: Select the top N most relevant documents to be used as context for the language model.
    Tools: LangChain Retrievers
5. Contextualization: Prepare the selected documents for input to the language model, ensuring that they are formatted appropriately.
    Tools: LangChain Retrievers 

### Augmentation for RAG
Augmentation for RAG involves providing the retrieved documents as context to the language model to guide the generation of responses. This process typically includes the following steps:
1. Context Preparation: Format the retrieved documents to be compatible with the input requirements of the language model. This may involve concatenating the documents, adding special tokens, or structuring the context in a specific way.
    Tools: LangChain Retrievers
2. Input Construction: Combine the user's query with the prepared context to create a single input for the language model.
    Tools: LangChain Retrievers
3. Context Injection: Ensure that the language model is aware of the context provided by the retrieved documents, allowing it to leverage this information during generation.
    Tools: LangChain LLMs
4. Prompt Engineering: Design prompts that effectively incorporate the context and guide the language model to generate accurate and relevant responses.
    Tools: LangChain Prompts
5. Input Validation: Verify that the constructed input meets the requirements of the language model, such as token limits or formatting constraints.
    Tools: LangChain LLMs
### Generation for RAG
Generation for RAG involves using the language model to generate a response based on the user's query and the provided context from the retrieved documents. This process typically includes the following steps:
1. Model Selection: Choose an appropriate language model that is capable of generating high-quality responses based on the input context.
    Tools: LangChain LLMs
2. Input Feeding: Provide the constructed input (query + context) to the language model for processing.
    Tools: LangChain LLMs       
3. Response Generation: Use the language model to generate a response that is informed by the context provided by the retrieved documents.
    Tools: LangChain LLMs
4. Post-processing: Clean and format the generated response to ensure clarity, coherence, and relevance.
    Tools: LangChain LLMs
5. Quality Assurance: Evaluate the quality of the generated response, checking for accuracy, relevance, and adherence to any specific requirements or guidelines.
    Tools: LangChain LLMs   

### Delivery for RAG
Delivery for RAG involves returning the generated response to the user in a clear and accessible manner. This process typically includes the following steps:
1. Response Formatting: Format the generated response to ensure it is easy to read and understand. This may involve adding headings, bullet points, or other formatting elements.
    Tools: LangChain LLMs
2. Output Integration: Integrate the response into the user interface or application where the user submitted the query.
    Tools: LangChain LLMs   
3. User Notification: Notify the user that their query has been processed and the response is ready.
    Tools: LangChain LLMs
4. Feedback Collection: Provide a mechanism for users to provide feedback on the quality and relevance of the response.
    Tools: LangChain LLMs
5. Continuous Improvement: Use user feedback to refine and improve the RAG system over time, ensuring that it continues to meet user needs and expectations.
    Tools: LangChain LLMs   



### RAG in Action

