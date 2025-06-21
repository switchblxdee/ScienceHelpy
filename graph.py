import logging
from pathlib import Path
from typing import Annotated, Any, List, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.message import AnyMessage
from langgraph.prebuilt import ToolNode, tools_condition

from config import settings
from model import GroqChatModel
from RAG import RAG_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TAVILY_API = settings.TAVILY_API


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rag_context: List[str]
    tool_used_bool: bool


def save_graph_png(graph: Any, filename: Path = Path("langgraph_workflow.png")) -> Path:
    """
    Сохраняет граф в PNG файл

    Параметры:
    - graph: скомпилированный граф LangGraph
    - filename: имя файла для сохранения (по умолчанию 'langgraph_workflow.png')
    """
    try:
        graph.get_graph().draw_mermaid_png(
            output_file_path=filename, background_color="white", padding=20
        )
        logger.info("Graph saved to %s", filename)
        return filename
    except Exception as ex:
        logger.error("Error saving graph: %s", ex)
        logger.info("Possible solutions:")
        logger.info("1. Install pyppeteer: pip install pyppeteer")
        logger.info("2. Install graphviz: pip install graphviz")
        raise


class Graph:
    def __init__(self):
        self._model = GroqChatModel()
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}
        self.tavily_tool = TavilySearchResults(tavily_api_key=TAVILY_API, max_results=3)
        self.tools = [self.tavily_tool]
        self.graph = self._build_graph(State)

    def _retrieve(self, state: State) -> dict[str, list[str]]:
        if not state["messages"]:
            logger.warning("No messages found in state.")
            return {"rag_context": []}

        last_message = state["messages"][-1].content
        return {"rag_context": RAG_answer(last_message)}

    def _generate(self, state: State) -> dict[str, list[AnyMessage]]:
        if not state["messages"]:
            logger.warning("No messages found in state.")
            return {"messages": []}
        user_prompt = state["messages"][-1].content

        rag_context = state["rag_context"][-1] if state["rag_context"] else ""

        response = self._model.generate(
            user_prompt=user_prompt, rag_context=rag_context
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

    def run(self, user_prompt: str) -> dict[str, Any]:
        return self.graph.invoke({"messages": [user_prompt]}, config=self.config)
