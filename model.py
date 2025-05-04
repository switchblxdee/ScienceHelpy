from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from tools import internet_search

from config import  MODEL_NAME, TOGETHER_API

class TogetherChatModel:
    def __init__(self, model_name: str = MODEL_NAME, api_key: str = TOGETHER_API):
        self.model_name = model_name
        self.api_key = api_key

        self._model = self._init_model()

    def _init_model(self):
        chat_model = ChatTogether(
            model=self.model_name,
            api_key=self.api_key,
            max_retries=2,
            max_tokens=1024,
            temperature=0.6,
            top_p=0.4,
        )

        chat_model = chat_model.bind_tools([internet_search])

        return chat_model
    
    def generate(self, user_prompt: str, rag_context: str):
        prompt = PromptTemplate(
            template="""
Ты - помощник в разборе статей по машинному обучению и ИИ. Отвечай на вопросы пользователя.

ВАЖНО: По умолчанию отвечай на вопросы самостоятельно, без использования инструментов. Используй инструмент internet_search ТОЛЬКО в следующих случаях:
1. Когда вопрос касается актуальных событий или новостей (например, последние достижения в ИИ)
2. Когда в предоставленном контексте нет информации для ответа
3. Когда вопрос выходит за рамки машинного обучения и ИИ

Пример 1:
Пользователь: Что такое машинное обучение?
Ассистент: Машинное обучение — это область искусственного интеллекта, которая...

Пример 2:
Пользователь: Кто выиграл чемпионат мира по шахматам в 2023?
Ассистент: [internet_search]

Пример 3:
Пользователь: Объясни принцип работы нейронных сетей
Ассистент: Нейронные сети — это вычислительные системы, которые...

Контекст: {rag_context}
Пользователь: {user_prompt}

Помни: Отвечай самостоятельно, если можешь! Используй поиск только в крайних случаях.
""",
input_variables=["rag_context", "user_prompt"]
        )

        chain = prompt | self._model
        
        response = chain.invoke({"rag_context": rag_context, "user_prompt": user_prompt})

        return response
    

    