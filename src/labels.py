"""Зрозумілі користувачу назви ознак і значень (для пояснень та сторінок)."""
import pandas as pd

# ознака: (назва, одиниця виміру або "")
FEATURES = {
    "Age": ("Вік", "р."),
    "Annual_Income": ("Річний дохід", ""),
    "Monthly_Inhand_Salary": ("Щомісячна зарплата «на руки»", ""),
    "Num_Bank_Accounts": ("Кількість банківських рахунків", ""),
    "Num_Credit_Card": ("Кількість кредитних карток", ""),
    "Interest_Rate": ("Відсоткова ставка", "%"),
    "Num_of_Loan": ("Кількість кредитів", ""),
    "Num_Loan_Types": ("Кількість видів кредитів", ""),
    "Delay_from_due_date": ("Середня затримка платежу", "дн."),
    "Num_of_Delayed_Payment": ("Кількість прострочених платежів", ""),
    "Changed_Credit_Limit": ("Зміна кредитного ліміту", "%"),
    "Num_Credit_Inquiries": ("Запити до кредитного бюро", ""),
    "Outstanding_Debt": ("Непогашений борг", ""),
    "Credit_Utilization_Ratio": ("Використання кредитного ліміту", "%"),
    "Credit_History_Age": ("Вік кредитної історії", ""),
    "Total_EMI_per_month": ("Щомісячні виплати за кредитами", ""),
    "Amount_invested_monthly": ("Щомісячні інвестиції", ""),
    "Monthly_Balance": ("Щомісячний залишок коштів", ""),
    "Debt_to_Income": ("Борг відносно річного доходу", ""),
    "EMI_to_Salary": ("Виплати за кредитами відносно доходу", ""),
    "Occupation": ("Професія", ""),
    "Credit_Mix": ("Структура кредитів", ""),
    "Payment_of_Min_Amount": ("Сплата лише мінімального платежу", ""),
    "Payment_Behaviour": ("Платіжна поведінка", ""),
}

OCCUPATIONS = {
    "Scientist": "Науковець", "Teacher": "Вчитель", "Engineer": "Інженер", "Entrepreneur": "Підприємець",
    "Developer": "Розробник", "Lawyer": "Юрист", "Media_Manager": "Медіаменеджер", "Doctor": "Лікар",
    "Journalist": "Журналіст", "Manager": "Менеджер", "Accountant": "Бухгалтер", "Musician": "Музикант",
    "Mechanic": "Механік", "Writer": "Письменник", "Architect": "Архітектор",
}
CATEGORY_VALUES = {
    "Credit_Mix": {"Good": "хороша", "Standard": "стандартна", "Bad": "погана"},
    "Payment_of_Min_Amount": {"Yes": "так", "No": "ні", "NM": "не вказано"},
    "Occupation": OCCUPATIONS,
}
_SIZE = {"Small": "малі", "Medium": "середні", "Large": "великі"}


def label(feature: str) -> str:
    return FEATURES.get(feature, (feature, ""))[0]


def _months(value: float) -> str:
    years, months = divmod(int(round(value)), 12)
    return f"{years} р. {months} міс." if years else f"{months} міс."


def display_value(feature: str, value) -> str:
    """Значення у зрозумілому вигляді: '10 дн.', '31.5%', 'хороша', '18 р. 4 міс.'"""
    if feature == "Payment_Behaviour":
        spend, _, size, *_ = str(value).split("_")
        return f"{'високі' if spend == 'High' else 'низькі'} витрати, {_SIZE.get(size, size)} платежі"
    if feature in CATEGORY_VALUES:
        return CATEGORY_VALUES[feature].get(str(value), str(value))
    if pd.isna(value):
        return "невідомо"
    if feature == "Credit_History_Age":
        return _months(float(value))
    if feature in ("Debt_to_Income", "EMI_to_Salary"):
        return f"{float(value):.2f}"
    unit = FEATURES.get(feature, ("", ""))[1]
    number = float(value)
    text = f"{number:,.0f}".replace(",", " ") if abs(number) >= 1000 or number == int(number) else f"{number:.1f}"
    return f"{text}{unit}" if unit == "%" else f"{text} {unit}".strip()
