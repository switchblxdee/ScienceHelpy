from model import GroqChatModel
from RAG import RAG_answer

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.message import AnyMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

from typing import List, TypedDict, Annotated

from config import TAVILY_API

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rag_context: List[str]
    tool_used_bool: bool

def save_graph_png(graph, filename='langgraph_workflow.png'):
    """
    Сохраняет граф в PNG файл
    
    Параметры:
    - graph: скомпилированный граф LangGraph
    - filename: имя файла для сохранения (по умолчанию 'langgraph_workflow.png')
    """
    try:
        png_data = graph.get_graph().draw_mermaid_png(
            output_file_path=filename,
            background_color='white',
            padding=20
        )
        print(f"Граф сохранен в {filename}")
        return filename
    
    except Exception as e:
        print(f"Ошибка сохранения графа: {e}")
        print("Возможные решения:")
        print("1. Установите pyppeteer: pip install pyppeteer")
        print("2. Установите graphviz: pip install graphviz")

class Graph:
    def __init__(self):
        self._model = GroqChatModel()
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}
        self.tavily_tool = TavilySearchResults(tavily_api_key=TAVILY_API, max_results=3)
        self.tools = [self.tavily_tool]

        self.graph = self._build_graph(State)
        
    def _retrieve(self, state: State):
        last_message = state["messages"][-1].content

        return {"rag_context": RAG_answer(last_message)}
    
    def _generate(self, state: State):
        messages = state["messages"]
        rag_context = state["rag_context"][-1] if state["rag_context"] else ""
        response = self._model.generate(
            user_prompt=messages,
            rag_context=rag_context
        )

        return {"messages": response}
    
    def _tool_node(self, state: State):

        return ToolNode(self.tools)
    
    def _build_graph(self, state: State):
        workflow = StateGraph(state)

        workflow.add_node("generate", self._generate)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("tools", self._tool_node)
    
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges(
            "generate",
            tools_condition,
        )
        workflow.add_edge("tools", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=self.memory)
    
    def run(self, user_prompt):
        # save_graph_png(self.graph)
        return self.graph.invoke(
            {"messages": [user_prompt]}, 
            config=self.config
        )