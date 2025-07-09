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
