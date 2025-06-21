import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    PATH_TO_PDFS_URL = os.getenv('PATH_TO_PDFS_URL')
    PATH_TO_PDFS = os.getenv('PATH_TO_PDFS')
    MODEL_NAME = os.getenv('MODEL_NAME')
    GROQ_API = os.getenv('GROQ_API')
    TAVILY_API = os.getenv('TAVILY_API')
else:
    Exception('The path to .env file is not exist')