import logging

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = settings.MODEL_NAME
GROQ_API = settings.GROQ_API
TAVILY_API = settings.TAVILY_API


class GroqChatModel:
    def __init__(self, model_name: str = MODEL_NAME, api_key: str = GROQ_API) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.tool = TavilySearchResults(tavily_api_key=TAVILY_API, max_results=3)
        self._model = self._init_model()

    def _init_model(self) -> ChatGroq:
        chat_model = ChatGroq(
            model=self.model_name,
            api_key=self.api_key,
            max_retries=2,
            max_tokens=2048,
            temperature=0.2,
        )

        chat_model = chat_model.bind_tools([self.tool])
        return chat_model

    def generate(self, user_prompt: str, rag_context: str):
        prompt = PromptTemplate(
            template="""
Ты - помощник для работы с научными материалами.

ВАЖНЫЕ ИНСТРУКЦИИ:
Используй инструменты

Дополнительный контекст: {rag_context}

Запрос пользователя: {user_prompt}
""",
            input_variables=["rag_context", "user_prompt"],
        )

        chain = prompt | self._model

        try:
            response = chain.invoke(
                {
                    "rag_context": rag_context,
                    "user_prompt": user_prompt,
                }
            )
            logger.info("Response generated successfully.")
            return response
        except Exception as ex:
            logger.error("Error generating response: %s", ex)
            raise RuntimeError(f"Failed to generate response: {ex}")
