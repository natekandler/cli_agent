import cmd
from libs.agent_runner import run_agent, display_graph_image
from langchain_core.prompts import ChatPromptTemplate
from libs.state_graph import StateGraphFactory

class CLIChat(cmd.Cmd):
    def __init__(self):
        primary_assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant tasked with answering a user's questions."
                    "You hae access to two tools: retrieve_documents and search_web."
                    "For any user questions about LLM agents, use the retrieve documents tool to get information for a vectorstore. Do not just summarize the article relevant to the question, use it to produce an answer to the question asked by the user. You should not say 'based on retrieved documents' in your answer."
                    "For any other questions, such as questions about current events, current weather, etc. use the search_web tool to get information from the web."
                ),
                ("placeholder", "{messages}")
            ]
        )
        self.graph = StateGraphFactory().create_graph(primary_assistant_prompt)
        super().__init__() 

    intro = "Hi I'm a chatbot. For help with commands, type 'help' or '?'. Otherwise just ask me a question. To finish the session type 'exit'\n"
    prompt = ">>> "


    def do_query(self, line):
       """Ask a question to the chatbot. The chatbot will process your input and return a response."""
       response = run_agent(line, self.graph)
       print(response.get('response', 'no response'))

    def do_show_graph(self, line):
        """Displays the graph used to determine which tool to call to answer your question."""
        display_graph_image(self.graph)

    def do_exit(self, line):
        """Exit the CLI"""
        print("Goodbye!")
        return True  
    
    def default(self, line):
        self.do_query(line)

if __name__ == "__main__":
    CLIChat().cmdloop()