# LangChain Models

**What to know**
    Langchain integrates with a variety of LLM providers (OpenAI, Cohere, Anthropic, Hugging Face, etc). Understand how to integrate, configure and chain models is crucial.

**Key Concepts**
    -   LLM Wrappers: ChatOpenAI, OpenAI, Anthropic, etc.
    -   Model I/O Interfaces: Unified APIs for different models.
    -   Temperature, max_tokens, stop sequence
    -   Callbacks and tracing for debugging.
    -   Streaming Outputs.
    -   Memory Integration.

**Interview Question**
1. What is difference between ChatOpenAI and OpenAI models in LangChain?
2. How do you configure temperature and other parameters in LangChain?
3. How would you stream tokens from LLM and use them in a real-time interface?
4. Can you explain how LangChain handles retries and timeouts with models?
5. How do you enable observability and debugging in LangChain?


**Production-Grade Example**
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

response = llm([HumanMessage(content="Summarize the last Apple WWDC event in bullet points.")])


**Insights**
- streaming=True is useful for live apps
- Adding callbacks allows real-time token-level control.
- Wrap it inside a LangChain chain or agent for advanced logic.
