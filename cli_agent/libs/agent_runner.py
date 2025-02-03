import uuid
import io
from PIL import Image
from langgraph.graph import StateGraph


def run_agent(input_text: str, graph: StateGraph):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    messages = graph.invoke({"messages": [("user", input_text)]}, config)
    
    return {"response": messages["messages"][-1].content, "messages": messages}


def display_graph_image(graph: StateGraph):
    img_bytes = graph.get_graph(xray=True).draw_mermaid_png() 
    img_buffer = io.BytesIO(img_bytes)
    
    img = Image.open(img_buffer)
    img.show()

