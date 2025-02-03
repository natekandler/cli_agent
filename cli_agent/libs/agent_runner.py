import io
from PIL import Image
from langgraph.graph import StateGraph


def run_agent(input_text: str, graph: StateGraph, thread_id: str):
    """
    Handles user queries while maintaining conversation history.
    """
    

    # Start with an empty history or retrieve existing one
    state = {
        "messages": [("user", input_text)],
        "conversation_history": [],  # This is now persistent!
    }
    updated_state = graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    
    return {
        "response": updated_state["messages"][-1].content,
        "conversation_history": updated_state["conversation_history"],  # Return updated history
    }


def display_graph_image(graph: StateGraph):
    img_bytes = graph.get_graph(xray=True).draw_mermaid_png() 
    img_buffer = io.BytesIO(img_bytes)
    
    img = Image.open(img_buffer)
    img.show()

