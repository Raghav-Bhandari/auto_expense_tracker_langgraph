import json
import asyncio
from typing import Annotated, TypedDict, Sequence
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from models import Expense, ParsedExpense, ExpenseInput
from expense_parser import ExpenseParser

class ExpenseState(TypedDict):
    messages: Sequence[BaseMessage]
    expense: ParsedExpense
    current_input: str
    complete: bool

def create_expense_workflow() -> StateGraph:
    workflow = StateGraph(ExpenseState)
    
    # Initialize parser
    parser = ExpenseParser()
    
    # Node for parsing expense from user input
    async def parse_expense(state: ExpenseState) -> ExpenseState:
        parsed = await parser.parse_expense(state["current_input"])
        state["expense"] = parsed
        return state
    
    # Node for checking if expense is complete
    async def check_completion(state: ExpenseState) -> dict:
        if state["expense"].is_complete():
            return {"next": "complete"}
        return {"next": "incomplete"}
    
    # Node for handling incomplete expenses
    async def handle_incomplete(state: ExpenseState) -> ExpenseState:
        prompt = parser.get_missing_info_prompt(state["expense"])
        state["messages"].append(HumanMessage(content=prompt))
        return state
    
    # Node for saving complete expense
    async def save_expense(state: ExpenseState) -> ExpenseState:
        expense = Expense(
            title=state["expense"].title,
            amount=state["expense"].amount,
            category=state["expense"].category
        )
        
        # Save to JSON file
        try:
            with open("expenses.json", "r") as f:
                expenses = json.load(f)
        except FileNotFoundError:
            expenses = []
        
        expenses.append(expense.model_dump())
        
        with open("expenses.json", "w") as f:
            json.dump(expenses, f, indent=2, default=str)
        
        state["complete"] = True
        return state

    # Add nodes to workflow
    workflow.add_node("parse", parse_expense)
    workflow.add_node("check", check_completion)
    workflow.add_node("incomplete", handle_incomplete)
    workflow.add_node("save", save_expense)

    # Define edges
    workflow.add_edge("parse", "check")
    workflow.add_conditional_edges(
        "check",
        lambda x: x["next"],
        {
            "complete": "save",
            "incomplete": "incomplete"
        }
    )

    workflow.add_edge("incomplete", END)
    workflow.add_edge("save", END)

    workflow.set_entry_point("parse")
    return workflow

async def process_expense(text: str) -> ExpenseState:
    workflow = create_expense_workflow()
    app = workflow.compile()
    state = ExpenseState(
        messages=[],
        expense=ParsedExpense(),
        current_input=text,
        complete=False
    )
    
    result = await app.ainvoke(state)
    return result

async def main():
    print("Welcome to the Expense Tracker!")
    print("Enter your expenses in natural language (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nExpense: ").strip()
            if user_input.lower() == 'quit':
                break
                
            result = await process_expense(user_input)
            
            if result["complete"]:
                print("✓ Expense recorded successfully!")
            elif result["messages"]:
                print(result["messages"][-1].content)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
