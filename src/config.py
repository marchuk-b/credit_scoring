from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
REPORT_PATH = MODELS_DIR / "report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "Credit_Score"
GROUP = "Customer_ID"
CLASSES = ["Poor", "Standard", "Good"]  # порядок = 0, 1, 2

# Ідентифікатори та поля без прогностичної цінності не потрапляють у модель
DROP_COLUMNS = ["ID", "Customer_ID", "Name", "SSN", "Month", "Type_of_Loan"]

# Допустимі діапазони: значення поза ними вважаються помилками даних -> NaN
VALID_RANGES = {
    "Age": (14, 100),
    "Annual_Income": (0, 1_000_000),
    "Monthly_Inhand_Salary": (0, 20_000),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (1, 40),
    "Num_of_Loan": (0, 12),
    "Delay_from_due_date": (-10, 80),
    "Num_of_Delayed_Payment": (0, 30),
    "Changed_Credit_Limit": (-20, 50),
    "Num_Credit_Inquiries": (0, 30),
    "Outstanding_Debt": (0, 10_000),
    "Credit_Utilization_Ratio": (0, 100),
    "Credit_History_Age": (0, 600),
    "Total_EMI_per_month": (0, 5_000),
    "Amount_invested_monthly": (0, 5_000),
    "Monthly_Balance": (0, 5_000),
}

CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
CATEGORY_VALUES = {
    "Credit_Mix": ["Bad", "Standard", "Good"],
    "Payment_of_Min_Amount": ["Yes", "No", "NM"],
}
DERIVED = ["Debt_to_Income", "EMI_to_Salary", "Num_Loan_Types"]
NUMERIC = list(VALID_RANGES) + DERIVED

# Скор у стилі FICO: 300 + 550 * (P(Good) + 0.5 * P(Standard))
SCORE_MIN, SCORE_RANGE = 300, 550
