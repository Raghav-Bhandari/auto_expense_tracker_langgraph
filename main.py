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
    context: str

def create_expense_workflow() -> StateGraph:
    workflow = StateGraph(ExpenseState)
    
    # Initialize parser
    parser = ExpenseParser()
    
    # Node for parsing expense from user input
    async def parse_expense(state: ExpenseState) -> ExpenseState:
        # Combine current input with context if available
        input_text = state["current_input"]
        if state["context"]:
            input_text = f"{state['context']} {input_text}"
        
        parsed = await parser.parse_expense(input_text)
        
        # Update existing expense with any new information
        current_expense = state["expense"]
        if parsed.title and not current_expense.title:
            current_expense.title = parsed.title
        if parsed.amount and not current_expense.amount:
            current_expense.amount = parsed.amount
        if parsed.category and not current_expense.category:
            current_expense.category = parsed.category
            
        state["expense"] = current_expense
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
        
        # Update context for next iteration
        context_parts = []
        if state["expense"].title:
            context_parts.append(f"title is {state['expense'].title}")
        if state["expense"].amount:
            context_parts.append(f"amount is {state['expense'].amount}")
        if state["expense"].category:
            context_parts.append(f"category is {state['expense'].category}")
        
        state["context"] = ", ".join(context_parts)
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
        except json.JSONDecodeError:
            expenses = []
        
        expenses.append(expense.model_dump())
        
        with open("expenses.json", "w") as f:
            json.dump(expenses, f, indent=2, default=str)
        
        state["complete"] = True
        state["messages"].append(HumanMessage(content="✓ Expense recorded successfully!"))
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

async def process_expense(text: str, previous_state: ExpenseState = None) -> ExpenseState:
    workflow = create_expense_workflow()
    app = workflow.compile()
    
    if previous_state and not previous_state["complete"]:
        # Continue with previous state if expense was incomplete
        previous_state["current_input"] = text
        state = previous_state
    else:
        # Start fresh
        state = ExpenseState(
            messages=[],
            expense=ParsedExpense(),
            current_input=text,
            complete=False,
            context=""
        )
    
    result = await app.ainvoke(state)
    return result

async def main():
    print("\n💰 Welcome to the Expense Tracker!")
    print("Enter your expenses in natural language (or 'quit' to exit)")
    print("Examples:")
    print("- 'spent 50 rs on lunch'")
    print("- 'movie ticket for 200 rupees'")
    print("- 'taxi ride home 150'")
    
    previous_state = None
    
    while True:
        try:
            user_input = input("\nExpense: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
                
            result = await process_expense(user_input, previous_state)
            
            if result["messages"]:
                print(result["messages"][-1].content)
            
            if not result["complete"]:
                previous_state = result
            else:
                previous_state = None
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            previous_state = None

if __name__ == "__main__":
    asyncio.run(main())
