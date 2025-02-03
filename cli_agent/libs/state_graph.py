import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from libs.llm import llm
from libs.document_retriever import retrieve_documents
from libs.web_search import search_web
from libs.assistant import Assistant, State

tools = [retrieve_documents, search_web]

class StateGraphFactory:
    def handle_tool_error(self, state: State) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes."
                )
                for tc in tool_calls
            ]
        }


    def create_tool_node_with_fallback(self, tools: list) -> dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )

    def create_graph(self, prompt: ChatPromptTemplate):
        assistant_runnable = prompt | llm.bind_tools(tools)

        builder = StateGraph(State)
        # Define nodes: these do the work
        assistant = Assistant(assistant_runnable)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", self.create_tool_node_with_fallback(tools))

        # Define edges: These determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assitatant is a tool call -> tool_condition routes to tools
            # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
            tools_condition
        )
        builder.add_edge("tools", "assistant")

        # The checkpointer lets the graph persist its state
        # sqlite3_conn = sqlite3.connect('checkpoints.sqlite')
        # sqlite3_memory_checkpoint = SqliteSaver(sqlite3_conn)
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)
