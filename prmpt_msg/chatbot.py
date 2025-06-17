from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  #who said what
import os

load_dotenv()

model = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),  #.igno
    base_url="https://openrouter.ai/api/v1"
)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)