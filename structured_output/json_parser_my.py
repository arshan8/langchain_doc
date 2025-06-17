from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. Load model
model = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 2. Setup parser
parser = JsonOutputParser()

# 3. Prompt template with parser instructions

template = PromptTemplate(
    template=(
        "Give me idea about event: {topic}.\n"      #does not enforce like pydantic
        "Include:\n"
        "- 'summary': a short paragraph\n"
        "- 'sentiment': one of ['positive', 'negative', 'neutral']\n"
        "- 'dates': list of important related dates (as strings)\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. Format the prompt (fixing the syntax error here)
prompt = template.invoke({"topic": "9/11"})

# 5. Invoke model
result = model.invoke(prompt, config={"max_tokens": 500})

# 6. Parse structured JSON output
structured_result = parser.parse(result.content)

# 7. Print result
print(structured_result)
