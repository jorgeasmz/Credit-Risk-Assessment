import pandas as pd
import io
import requests

def load_data():
    """
    Fetches the German Credit Data from the UCI Machine Learning Repository.
    
    The dataset contains 1000 entries with 20 categorical/symbolic attributes.
    Original Data Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the raw data with readable column headers.
    """
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    columns = [
        'checkin_acc',       # Status of existing checking account
        'duration',          # Duration in month
        'credit_history',    # Credit history
        'purpose',           # Purpose of the loan
        'amount',            # Credit amount
        'savings_acc',       # Savings account/bonds
        'present_emp_since', # Present employment since
        'installment_rate',  # Installment rate in percentage of disposable income
        'personal_status',   # Personal status and sex
        'other_debtors',     # Other debtors / guarantors
        'residing_since',    # Present residence since
        'property',          # Property
        'age',               # Age in years
        'inst_plans',        # Other installment plans
        'housing',           # Housing
        'num_credits',       # Number of existing credits at this bank
        'job',               # Job
        'dependents',        # Number of people being liable to provide maintenance for
        'telephone',         # Telephone
        'foreign_worker',    # Foreign worker
        'status'             # Cost Matrix (Risk)
    ]

    try:
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        response.raise_for_status() # Check for HTTP errors
        
        # Load string data into pandas. Using ' ' separator as per dataset format.
        df = pd.read_csv(io.StringIO(response.text), sep=' ', names=columns)
        
        # Preprocessing the Target Variable:
        # In the original dataset: 1 = Good, 2 = Bad.
        # So we map: 2 -> 1 (Risk/Bad), 1 -> 0 (No Risk/Good).
        df['status'] = df['status'].map({1: 0, 2: 1})
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise