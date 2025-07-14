What are Prompts?

Static Vs Dynamic Prompts

    -   Static Prompts: Always user defined the prompt
    -   Dynamic Prompts: We ask user to set some properties and prompt will be generated based upon the template provided.

Prompt Template
    A PromptTemplate in LangChain is a structured way to create prompts dynamically by inserting variables into a predefined template. Instead of hardcording prompts, PromptTemplate allow you to define placeholders that cn be filled in at runtime with different inputs.

    This makes it reusable, flexible and easy to manage, especially when working with dynamic user inputs or automted workflows.

    Why Use PromptTemplate over f String?
    1. Default Validation: If user provided extra properties or missed adding any property.
    2. Reusable: Can be reused by multiple application.
    3. LangChain Ecosystem: Create chain and invoke once for all.



Messages:
1. System Message
2. Human Message
3. AI Message

Invoke
    - Single Message
        -   Static Message
        -   Dynamic Message(Prompt Template)
    
    - List of Messages(multi-turn conversation)
        -   Static Message(System Message,Human Message, AI Message)
        -   Dynamic Message


Prompt Template and Chat Prompt Template has same work internally.
The only different is Prompt template is used for single turn message and Chat Prompt Template is used for Multi turn conversation



Message Place Holder
A message place holder in LangChain is a special placeholder used inside  ChatPrompt Template to dynamically insert chat history or a list of messages at runtime.



🔍 **What to Know**
LangChain separates prompt templates from the model. This modularity helps in reusability, evaluation, and A/B testing.

📌 **Key Concepts**
-   PromptTemplate and ChatPromptTemplate
-   Dynamic prompts using input variables
-   Few-shot prompting
-   Prompt tuning and testing
-   Best practices (clarity, structure, verbosity, etc.)

❓ **Interview Questions**
1. What’s the difference between PromptTemplate and ChatPromptTemplate?
2. How do you insert dynamic variables in a prompt?
3. How do you do few-shot prompting in LangChain?
4. How do you evaluate prompt quality and iterate over it?
5. How can prompt templates help in production AI applications?

💡 **Production-Grade Example**
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "You are an expert legal assistant. Extract the key clauses from the following legal contract:\n\n{contract_text}"
)

formatted_prompt = prompt.format_messages(contract_text="This Agreement is made on...")
response = llm(formatted_prompt)



🔑 **Insights:**

-   Use .format_messages() with ChatPromptTemplate for chat-specific formatting.
-   Great for reusable prompt components.
-   Combine this with PromptLayer or LangSmith to A/B test prompts.