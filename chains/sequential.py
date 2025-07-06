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
prompt2 = PromptTemplate(
    template='smmarize the {prev_output} in 1 line',
    input_variables=['prev_output']
)

model = llm

parser = StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

