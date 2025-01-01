from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Expense(BaseModel):
    title: str
    amount: float
    category: str
    date: datetime = Field(default_factory=datetime.now)

class ExpenseInput(BaseModel):
    text: str

class ParsedExpense(BaseModel):
    title: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    
    def is_complete(self) -> bool:
        return all([self.title, self.amount, self.category])

    def get_missing_fields(self) -> list[str]:
        missing = []
        if not self.title:
            missing.append("title")
        if not self.amount:
            missing.append("amount")
        if not self.category:
            missing.append("category")
        return missing
