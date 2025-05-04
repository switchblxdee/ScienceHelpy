from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate

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
            temperature=0.8,
            top_p=0.8,
        )

        return chat_model
    
    def generate(self, user_prompt: str, rag_context: str):
        prompt = PromptTemplate(
            template="""
Ты - помощник в разборе статей по мащинному обучению и искуслвенному интеллекту. Отвечай на вопрсоы пользователя, чтобы он разбирался, что написано в статье. Помни, надо именно помочь, а не выдавать ответы.
Контекст: {rag_context}
Пользователь: {user_prompt}
""",
input_variables=["rag_context", "user_prompt"]
        )

        chain = prompt | self._model
        
        response = chain.invoke({"rag_context": rag_context, "user_prompt": user_prompt})

        return response
    

    