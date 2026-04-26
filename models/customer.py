from typing import Literal
from pydantic import BaseModel

#This class is for customer information
class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    mobile: str
    arpu_monthly: float
    arpu_tier: Literal["black", "postpaid_high", "postpaid_std", "prepaid_annual", "prepaid_monthly"]
    tenure_years: float
    plan_type: Literal["postpaid", "prepaid"]
    plan_name: str
    location: str
