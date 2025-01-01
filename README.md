# Auto Expense Tracker

A natural language expense tracking application that uses LangChain and LangGraph to process and record expenses.

## Features

- Natural language expense input processing
- Automatic extraction of expense title, amount, and category
- Interactive prompts for missing information
- JSON-based expense storage
- Modular and extensible design

## Setup

1. Clone the repository
2. Create and activate a conda environment:
```bash
conda create -n langgraph_agent python=3.12
conda activate langgraph_agent
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

Run the application:
```bash
python main.py
```

Example inputs:
- "I spent $20 on lunch today"
- "Bought groceries for $50"
- "Taxi ride to work"

The application will:
1. Parse your natural language input
2. Extract expense details (title, amount, category)
3. Prompt for any missing information
4. Save the complete expense to expenses.json

## Project Structure

- `main.py`: Main application with LangGraph workflow
- `models.py`: Pydantic models for data validation
- `expense_parser.py`: Natural language parsing logic
- `expenses.json`: Storage for recorded expenses

## Requirements

- Python 3.12
- OpenAI API key
- Dependencies listed in requirements.txt
