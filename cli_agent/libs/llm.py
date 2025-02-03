from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='llama3-groq-tool-use',
    temperature=0
)
