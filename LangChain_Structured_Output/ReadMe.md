Structured Output


Text - >  LLM -> Ouput Text 
This is an example of unstructured output.


In LangChain, structured output refers to the practice of having language models return responses in a well defined data format(e.g. JSON), rather than free-form text. This makes the model output easier to parse and work with programmatically.


Why do we need Structured Output
1. Data Extraction
2. API Building
3. Agents



Output Parser is a technique that will parse unstructured output to generate structured output.

with_structured_output

Data_format following ways to genereate
1. Typed Dict
    TypedDict is a way to define a dictionary in Python where you specify what keys and values should exists. It helps ensure that your dictionary follows a specific structure.

    Why use TypedDict?
        -   It tells Python what keys are required and what types of values they should have.
        -   It doesnot validate data at runtime(it just helps with type hints for better coding.)

2. Pydantic
3. JSON_Schema