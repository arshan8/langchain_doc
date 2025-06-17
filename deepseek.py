from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),  #.igno
    base_url="https://openrouter.ai/api/v1"
)

result = model.invoke('tell me about bmsce')
print(result.content)