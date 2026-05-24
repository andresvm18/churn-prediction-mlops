from pydantic import BaseModel

# Define the request body schema for customer data
# Pydantic automatically validates data types and required fields
class CustomerData(BaseModel):
    gender: str                          # Male/Female
    senior_citizen: int                  # 0 or 1 (No/Yes)
    partner: str                         # Yes/No
    dependents: str                      # Yes/No
    tenure_months: int                   # Months as customer
    phone_service: str                   # Yes/No
    multiple_lines: str                  # Yes/No/No phone service
    internet_service: str                # DSL/Fiber optic/No
    online_security: str                 # Yes/No/No internet service
    online_backup: str                   # Yes/No/No internet service
    device_protection: str               # Yes/No/No internet service
    tech_support: str                    # Yes/No/No internet service
    streaming_tv: str                    # Yes/No/No internet service
    streaming_movies: str                # Yes/No/No internet service
    contract: str                        # Month-to-month/One year/Two year
    paperless_billing: str               # Yes/No
    payment_method: str                  # Electronic check/Mailed check/Bank transfer/Credit card
    monthly_charges: float               # Monthly bill amount
    total_charges: float                 # Lifetime total charges