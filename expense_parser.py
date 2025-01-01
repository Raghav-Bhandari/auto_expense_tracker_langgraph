from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from models import ParsedExpense
from dotenv import load_dotenv
import os

EXPENSE_PROMPT = """Extract expense information from the following text. If any information is missing, leave those fields as null.
The output should be in JSON format with the following fields:
- title: The item or service purchased
- amount: The cost as a number (without currency symbols)
- category: The type of expense (e.g., food, transport, entertainment)

Text: {input_text}

Only respond with the JSON, no additional text.
"""

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class ExpenseParser:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(EXPENSE_PROMPT)
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        self.output_parser = JsonOutputParser()
        
        self.chain = self.prompt | self.model | self.output_parser

    async def parse_expense(self, text: str) -> ParsedExpense:
        try:
            result = await self.chain.ainvoke({"input_text": text})
            return ParsedExpense(
                title=result.get("title"),
                amount=result.get("amount"),
                category=result.get("category")
            )
        except Exception as e:
            print(f"Error parsing expense: {e}")
            return ParsedExpense()

    def get_missing_info_prompt(self, parsed_expense: ParsedExpense) -> str:
        missing = parsed_expense.get_missing_fields()
        if not missing:
            return ""
        
        fields_str = " and ".join(missing)
        current_info = []
        
        if parsed_expense.title:
            current_info.append(f"title: {parsed_expense.title}")
        if parsed_expense.amount:
            current_info.append(f"amount: {parsed_expense.amount}")
        if parsed_expense.category:
            current_info.append(f"category: {parsed_expense.category}")
            
        current_str = ", ".join(current_info) if current_info else "no information"
        
        return f"Please provide the {fields_str} for the expense ({current_str}):"
