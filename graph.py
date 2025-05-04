from model import TogetherChatModel
from RAG import RAG_answer

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated, List

class State(TypedDict):
    messages: Annotated[list, add_messages]
    rag_context: List[str]
    use_internet_search: bool

class GraphLangGraph:
    def __init__(self):
        self._model = TogetherChatModel()
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}

        self.graph = self._build_graph(State)

    def _retrieve(self, state: State):
        last_message = state["messages"][-1].content
        
        return {"rag_context": RAG_answer(last_message)}

    def _generate(self, state: State):
        messages = state["messages"]
        rag_context = state["rag_context"][-1] if state["rag_context"] else ""
        response = self._model.generate(rag_context=rag_context, user_prompt=messages)
        return {"messages": response}

    def _build_graph(self, state: State):
        workflow = StateGraph(state)
        
        workflow.add_node("rag", self._retrieve)
        workflow.add_node("generate", self._generate)
        
        workflow.add_edge(START, "rag")
        workflow.add_edge("rag", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=self.memory)
    
    def generate(self, user_prompt):
        graph_image = self.graph.get_graph().draw_mermaid_png()

        with open("langgraph_workflow.png", "wb") as f:
            f.write(graph_image)

        response = self.graph.invoke({"messages": user_prompt}, config=self.config)
        print(response["messages"][-1].tool_calls)

        return {"messages": response["messages"][-1].content}
