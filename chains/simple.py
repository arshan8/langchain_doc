from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Loads the .env file where your OPENAI_API_KEY is stored

llm = ChatOpenAI(
    model="deepseek/deepseek-r1-0528:free",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = llm

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()