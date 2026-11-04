"""Перевірка вхідних даних API."""
ALLOWED = {
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
    "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
}
REQUIRED = {"Age", "Annual_Income", "Outstanding_Debt", "Delay_from_due_date"}


def validate(payload):
    if not isinstance(payload, dict):
        return None, "Очікується JSON-об'єкт"
    data = {k: v for k, v in payload.items() if k in ALLOWED and v not in ("", None)}
    missing = sorted(REQUIRED - data.keys())
    if missing:
        return None, f"Відсутні обов'язкові поля: {', '.join(missing)}"
    return data, None
