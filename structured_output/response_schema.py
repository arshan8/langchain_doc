from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


# If you're doing strict structured tasks (like filling forms, extracting fields) — use StructuredOutputParser.
# If you just want quick JSON parsing from a model that behaves well — JsonOutputParser is enough.

load_dotenv()

# 1. Load model
model = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),   #no data validation
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# 4. Format the prompt (fixing the syntax error here)
prompt = template.invoke({"topic": "9/11"})

# 5. Invoke model
result = model.invoke(prompt, config={"max_tokens": 500})

# 6. Parse structured JSON output
structured_result = parser.parse(result.content)

# 7. Print result
print(structured_result)
