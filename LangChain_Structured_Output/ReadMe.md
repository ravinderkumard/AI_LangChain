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

    1. Simple TypedDict
    2. Annotated TypedDict
    3. Literal
    4. More Complex -> with pros and cons

2. Pydantic
    Pydantic is a data validation and data parsing library for Python. It ensures that the data you work with is correct, structured and type-safe
        pydantic will perform type conversion as applicable
        Validator for email using EmailStr, This will throw error 
            raise ImportError('email-validator is not installed, run `pip install pydantic[email]`') from e
            ImportError: email-validator is not installed, run `pip install pydantic[email]`

            pip3 install email-validator
            This will install email-validator

            pydantic_core._pydantic_core.ValidationError: 1 validation error for Student
                email
                value is not a valid email address: An email address must have an @-sign. [type=value_error, input_value='ab', input_type=str]

            Now if you don't provide a valid email, it will throw an error.

            Field Function:
                Set Default value, constraints, descritpion, regex expression

               | Situation                                                      | Does Pydantic run its type–casting pipeline?                           | Result for `int → float`                        |
| -------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------- |
| **Class‑level default**<br>`cgpa: float = Field(default=6, …)` | **No** (defaults are trusted unless you explicitly ask for validation) | Stored exactly as you wrote it → `6` (an `int`) |
| **Runtime input**<br>`Student(**{"cgpa": 8})`                  | **Yes** (all incoming data are validated & coerced)                    | Cast to `6.0` / `8.0` (a `float`)               |

                Why this design?
                Performance:
                Validating every class‑level default each time a model is created would slow things down. Pydantic assumes the developer wrote correct defaults.

                Safety for external data:
                Data coming from outside the program (API payloads, DB rows, user input) is untrusted, so Pydantic validates and coerces it.


            How to make defaults go through validation (and get 6.0)
                Pydantic v1.x
                python
                Copy
                Edit
                from pydantic import BaseModel, Field

                class Student(BaseModel):
                    class Config:
                        validate_default = True        # validate all defaults

                    cgpa: float = Field(6, ge=0.0, le=10.0)

                print(Student())        # cgpa=6.0
                Pydantic v2.x
                python
                Copy
                Edit
                from pydantic import BaseModel, Field, ConfigDict

                class Student(BaseModel):
                    model_config = ConfigDict(validate_default=True)

                    cgpa: float = Field(6, ge=0.0, le=10.0)

                print(Student())        # cgpa=6.0
                You can also just write the default as a float (6.0), or make the field strict:

                python
                Copy
                Edit
                cgpa: float = Field(6, strict=True)  # raises error unless value is already a float
                So:

                Default value 6 → trusted, kept as‑is (int)

                Input value 8 → validated, coerced to 8.0 (float)

                Turn on validate_default (or provide the default as a float) if you want consistent float representations everywhere.

3. JSON_Schema
    Used while working with multiple language.


# When to use what?
1. Use TypedDict if:
    -   you only need type hints(basic structure enforcement)
    -   your don't need validation(e.g. checking number are positive)
    -   you trust llm to return correct data.

2. Use Pydantic if:
    -   You need data validation(e.g. sentiment must be "positive", "neutral", "negative")
    -   you need default values if the LLM misses fields.
    -   you want automatic type conversion ("100" -> 100)

3. Use JSON Schema if:
    -   You don't want to import extra Python libraries.
    -   You need validation but don't need Python objects.
    -   you want to define structure in a standard JSON format.





 | Feature        | TypedDict | Pydantic   | JSON Schema |
| ---------------- | --------------------- | -------------- | -------------- |
| **Basic structure** | **Yes** | **Yes** | **Yes** |
| **Typed enforcement**    | **Yes**  | **Yes** | **Yes** |
| **Data validation**    | **No**  | **Yes** | **Yes** |
| **Data Values**    | **No**  | **Yes** | **No** |
| **Automatic conversion**    | **No**  | **Yes** | **No** |
| **Cross-language compatibility**    | **No**  | **No** | **Yes** |
| ---------------- | --------------------- | -------------- | -------------- |