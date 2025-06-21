from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from config import  MODEL_NAME, GROQ_API, TAVILY_API

class GroqChatModel:
    def __init__(self, model_name: str = MODEL_NAME, api_key: str = GROQ_API):
        self.model_name = model_name
        self.api_key = api_key
        self.tool = TavilySearchResults(tavily_api_key=TAVILY_API, max_results=3)
        self._model = self._init_model()

    def _init_model(self):
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
input_variables=["rag_context", "user_prompt"]
        )

        chain = prompt | self._model
        
        response = chain.invoke({"rag_context": rag_context, "user_prompt": user_prompt})

        return response
