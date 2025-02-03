from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable =  runnable

    def __call__(self, state: State, config: RunnableConfig):
        """
        Call method to invoke the LLM and handle it's responses. 
        Re-prompt the assistant if the reponse is not a tool call or meaningful text.

        Args:
            state (State): The current state containing messages.
            config (RunnableConfig): the configuration for the runnable.

        Returns:
        dict: the final state containing the updates messages:

        """
        while True:
            result = self.runnable.invoke(state) # invoke the LLM

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}