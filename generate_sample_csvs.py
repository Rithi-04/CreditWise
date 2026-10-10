import os
import numpy as np
import pandas as pd

FIRST_NAMES = [
    "Eleanor", "Marcus", "Sarah", "David", "Amara", "Carlos", "Aisha", "Liam",
    "Sophia", "Vikram", "Elena", "Chen", "Maya", "Jackson", "Fatima", "Oliver",
    "Priya", "Lucas", "Zoe", "Dmitri", "Hannah", "Gabriel", "Leila", "Mateo"
]
LAST_NAMES = [
    "Vance", "Chen", "Jenkins", "Kim", "O'Connor", "Rodriguez", "Patel", "Wright",
    "Al-Mansoor", "Soderberg", "Gupta", "Wong", "Lin", "Taylor", "Kowalski", "Abadi",
    "Nakamura", "Dubois", "Santos", "Morales", "Novak", "Olsen", "Takahashi", "Zhao"
]

def generate_names(n_samples: int, seed: int) -> list:
    np.random.seed(seed)
    names = []
    for i in range(n_samples):
        first = FIRST_NAMES[np.random.randint(0, len(FIRST_NAMES))]
        last = LAST_NAMES[np.random.randint(0, len(LAST_NAMES))]
        names.append(f"{first} {last}")
    return names

def create_high_risk_portfolio(n_samples=50, start_id=1001, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    limit_bal = np.random.choice([10000, 20000, 30000, 50000], size=n_samples)
    sex = np.random.choice([1, 2], size=n_samples)
    education = np.random.choice([1, 2, 3], size=n_samples)
    marriage = np.random.choice([1, 2], size=n_samples)
    age = np.random.randint(22, 60, size=n_samples)
    
    pay_0 = np.random.choice([2, 3, 4, 5, 6], size=n_samples)
    pay_2 = np.random.choice([1, 2, 3], size=n_samples)
    pay_3 = np.random.choice([0, 1, 2], size=n_samples)
    pay_4 = np.random.choice([0, 1, 2], size=n_samples)
    pay_5 = np.random.choice([0, 1, 2], size=n_samples)
    pay_6 = np.random.choice([0, 1, 2], size=n_samples)
    
    bill_amt1 = limit_bal * np.random.uniform(0.85, 1.1, size=n_samples)
    bill_amt2 = bill_amt1 * np.random.uniform(0.9, 1.0, size=n_samples)
    bill_amt3 = bill_amt2 * np.random.uniform(0.9, 1.0, size=n_samples)
    bill_amt4 = bill_amt3 * np.random.uniform(0.9, 1.0, size=n_samples)
    bill_amt5 = bill_amt4 * np.random.uniform(0.9, 1.0, size=n_samples)
    bill_amt6 = bill_amt5 * np.random.uniform(0.9, 1.0, size=n_samples)
    
    pay_amt1 = bill_amt1 * np.random.uniform(0.0, 0.05, size=n_samples)
    pay_amt2 = bill_amt2 * np.random.uniform(0.0, 0.05, size=n_samples)
    pay_amt3 = bill_amt3 * np.random.uniform(0.0, 0.05, size=n_samples)
    pay_amt4 = bill_amt4 * np.random.uniform(0.0, 0.05, size=n_samples)
    pay_amt5 = bill_amt5 * np.random.uniform(0.0, 0.05, size=n_samples)
    pay_amt6 = bill_amt6 * np.random.uniform(0.0, 0.05, size=n_samples)
    
    default_target = np.random.choice([1, 0], size=n_samples, p=[0.85, 0.15])
    
    df = pd.DataFrame({
        'ID': range(start_id, start_id + n_samples),
        'customer_name': generate_names(n_samples, seed),
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': np.round(bill_amt1, 2), 'BILL_AMT2': np.round(bill_amt2, 2),
        'BILL_AMT3': np.round(bill_amt3, 2), 'BILL_AMT4': np.round(bill_amt4, 2),
        'BILL_AMT5': np.round(bill_amt5, 2), 'BILL_AMT6': np.round(bill_amt6, 2),
        'PAY_AMT1': np.round(pay_amt1, 2), 'PAY_AMT2': np.round(pay_amt2, 2),
        'PAY_AMT3': np.round(pay_amt3, 2), 'PAY_AMT4': np.round(pay_amt4, 2),
        'PAY_AMT5': np.round(pay_amt5, 2), 'PAY_AMT6': np.round(pay_amt6, 2),
        'default.payment.next.month': default_target
    })
    return df

def create_medium_risk_portfolio(n_samples=50, start_id=2001, seed=43) -> pd.DataFrame:
    np.random.seed(seed)
    limit_bal = np.random.choice([50000, 80000, 100000, 150000], size=n_samples)
    sex = np.random.choice([1, 2], size=n_samples)
    education = np.random.choice([1, 2, 3], size=n_samples)
    marriage = np.random.choice([1, 2], size=n_samples)
    age = np.random.randint(25, 65, size=n_samples)
    
    pay_0 = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
    pay_2 = np.random.choice([-1, 0, 1], size=n_samples)
    pay_3 = np.random.choice([-1, 0, 1], size=n_samples)
    pay_4 = np.random.choice([-1, 0], size=n_samples)
    pay_5 = np.random.choice([-1, 0], size=n_samples)
    pay_6 = np.random.choice([-1, 0], size=n_samples)
    
    bill_amt1 = limit_bal * np.random.uniform(0.4, 0.75, size=n_samples)
    bill_amt2 = bill_amt1 * np.random.uniform(0.8, 1.0, size=n_samples)
    bill_amt3 = bill_amt2 * np.random.uniform(0.8, 1.0, size=n_samples)
    bill_amt4 = bill_amt3 * np.random.uniform(0.8, 1.0, size=n_samples)
    bill_amt5 = bill_amt4 * np.random.uniform(0.8, 1.0, size=n_samples)
    bill_amt6 = bill_amt5 * np.random.uniform(0.8, 1.0, size=n_samples)
    
    pay_amt1 = bill_amt1 * np.random.uniform(0.05, 0.2, size=n_samples)
    pay_amt2 = bill_amt2 * np.random.uniform(0.05, 0.2, size=n_samples)
    pay_amt3 = bill_amt3 * np.random.uniform(0.05, 0.2, size=n_samples)
    pay_amt4 = bill_amt4 * np.random.uniform(0.05, 0.2, size=n_samples)
    pay_amt5 = bill_amt5 * np.random.uniform(0.05, 0.2, size=n_samples)
    pay_amt6 = bill_amt6 * np.random.uniform(0.05, 0.2, size=n_samples)
    
    default_target = np.random.choice([1, 0], size=n_samples, p=[0.4, 0.6])
    
    df = pd.DataFrame({
        'ID': range(start_id, start_id + n_samples),
        'customer_name': generate_names(n_samples, seed),
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': np.round(bill_amt1, 2), 'BILL_AMT2': np.round(bill_amt2, 2),
        'BILL_AMT3': np.round(bill_amt3, 2), 'BILL_AMT4': np.round(bill_amt4, 2),
        'BILL_AMT5': np.round(bill_amt5, 2), 'BILL_AMT6': np.round(bill_amt6, 2),
        'PAY_AMT1': np.round(pay_amt1, 2), 'PAY_AMT2': np.round(pay_amt2, 2),
        'PAY_AMT3': np.round(pay_amt3, 2), 'PAY_AMT4': np.round(pay_amt4, 2),
        'PAY_AMT5': np.round(pay_amt5, 2), 'PAY_AMT6': np.round(pay_amt6, 2),
        'default.payment.next.month': default_target
    })
    return df

def create_low_risk_portfolio(n_samples=50, start_id=3001, seed=44) -> pd.DataFrame:
    np.random.seed(seed)
    limit_bal = np.random.choice([100000, 200000, 300000, 500000], size=n_samples)
    sex = np.random.choice([1, 2], size=n_samples)
    education = np.random.choice([1, 2], size=n_samples)
    marriage = np.random.choice([1, 2], size=n_samples)
    age = np.random.randint(28, 68, size=n_samples)
    
    pay_0 = np.random.choice([-2, -1, 0], size=n_samples, p=[0.3, 0.4, 0.3])
    pay_2 = np.random.choice([-2, -1, 0], size=n_samples)
    pay_3 = np.random.choice([-2, -1, 0], size=n_samples)
    pay_4 = np.random.choice([-2, -1, 0], size=n_samples)
    pay_5 = np.random.choice([-2, -1, 0], size=n_samples)
    pay_6 = np.random.choice([-2, -1, 0], size=n_samples)
    
    bill_amt1 = limit_bal * np.random.uniform(0.05, 0.3, size=n_samples)
    bill_amt2 = bill_amt1 * np.random.uniform(0.5, 1.0, size=n_samples)
    bill_amt3 = bill_amt2 * np.random.uniform(0.5, 1.0, size=n_samples)
    bill_amt4 = bill_amt3 * np.random.uniform(0.5, 1.0, size=n_samples)
    bill_amt5 = bill_amt4 * np.random.uniform(0.5, 1.0, size=n_samples)
    bill_amt6 = bill_amt5 * np.random.uniform(0.5, 1.0, size=n_samples)
    
    pay_amt1 = bill_amt1 * np.random.uniform(0.5, 1.0, size=n_samples)
    pay_amt2 = bill_amt2 * np.random.uniform(0.5, 1.0, size=n_samples)
    pay_amt3 = bill_amt3 * np.random.uniform(0.5, 1.0, size=n_samples)
    pay_amt4 = bill_amt4 * np.random.uniform(0.5, 1.0, size=n_samples)
    pay_amt5 = bill_amt5 * np.random.uniform(0.5, 1.0, size=n_samples)
    pay_amt6 = bill_amt6 * np.random.uniform(0.5, 1.0, size=n_samples)
    
    default_target = np.random.choice([1, 0], size=n_samples, p=[0.04, 0.96])
    
    df = pd.DataFrame({
        'ID': range(start_id, start_id + n_samples),
        'customer_name': generate_names(n_samples, seed),
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': np.round(bill_amt1, 2), 'BILL_AMT2': np.round(bill_amt2, 2),
        'BILL_AMT3': np.round(bill_amt3, 2), 'BILL_AMT4': np.round(bill_amt4, 2),
        'BILL_AMT5': np.round(bill_amt5, 2), 'BILL_AMT6': np.round(bill_amt6, 2),
        'PAY_AMT1': np.round(pay_amt1, 2), 'PAY_AMT2': np.round(pay_amt2, 2),
        'PAY_AMT3': np.round(pay_amt3, 2), 'PAY_AMT4': np.round(pay_amt4, 2),
        'PAY_AMT5': np.round(pay_amt5, 2), 'PAY_AMT6': np.round(pay_amt6, 2),
        'default.payment.next.month': default_target
    })
    return df

def create_mixed_batch_alpha(n_samples=100, start_id=4001, seed=45) -> pd.DataFrame:
    np.random.seed(seed)
    limit_bal = np.random.uniform(20000, 400000, n_samples)
    sex = np.random.choice([1, 2], n_samples)
    education = np.random.choice([1, 2, 3, 4], n_samples, p=[0.35, 0.45, 0.15, 0.05])
    marriage = np.random.choice([1, 2, 3], n_samples, p=[0.45, 0.50, 0.05])
    age = np.random.randint(21, 72, n_samples)
    
    pay_0 = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.2, 0.55, 0.1, 0.1, 0.05])
    pay_2 = np.random.choice([-1, 0, 1, 2], n_samples, p=[0.2, 0.6, 0.1, 0.1])
    pay_3 = np.random.choice([-1, 0, 1, 2], n_samples, p=[0.2, 0.6, 0.1, 0.1])
    pay_4 = np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.7, 0.1])
    pay_5 = np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.7, 0.1])
    pay_6 = np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.7, 0.1])
    
    bill_amt1 = limit_bal * np.random.uniform(0.1, 0.9, n_samples)
    bill_amt2 = bill_amt1 * np.random.uniform(0.7, 1.1, n_samples)
    bill_amt3 = bill_amt2 * np.random.uniform(0.7, 1.1, n_samples)
    bill_amt4 = bill_amt3 * np.random.uniform(0.7, 1.1, n_samples)
    bill_amt5 = bill_amt4 * np.random.uniform(0.7, 1.1, n_samples)
    bill_amt6 = bill_amt5 * np.random.uniform(0.7, 1.1, n_samples)
    
    pay_amt1 = bill_amt1 * np.random.uniform(0.02, 0.5, n_samples)
    pay_amt2 = bill_amt2 * np.random.uniform(0.02, 0.5, n_samples)
    pay_amt3 = bill_amt3 * np.random.uniform(0.02, 0.5, n_samples)
    pay_amt4 = bill_amt4 * np.random.uniform(0.02, 0.5, n_samples)
    pay_amt5 = bill_amt5 * np.random.uniform(0.02, 0.5, n_samples)
    pay_amt6 = bill_amt6 * np.random.uniform(0.02, 0.5, n_samples)
    
    default_target = np.random.choice([1, 0], size=n_samples, p=[0.22, 0.78])
    
    df = pd.DataFrame({
        'ID': range(start_id, start_id + n_samples),
        'customer_name': generate_names(n_samples, seed),
        'LIMIT_BAL': np.round(limit_bal, -3),
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': np.round(bill_amt1, 2), 'BILL_AMT2': np.round(bill_amt2, 2),
        'BILL_AMT3': np.round(bill_amt3, 2), 'BILL_AMT4': np.round(bill_amt4, 2),
        'BILL_AMT5': np.round(bill_amt5, 2), 'BILL_AMT6': np.round(bill_amt6, 2),
        'PAY_AMT1': np.round(pay_amt1, 2), 'PAY_AMT2': np.round(pay_amt2, 2),
        'PAY_AMT3': np.round(pay_amt3, 2), 'PAY_AMT4': np.round(pay_amt4, 2),
        'PAY_AMT5': np.round(pay_amt5, 2), 'PAY_AMT6': np.round(pay_amt6, 2),
        'default.payment.next.month': default_target
    })
    return df

def create_edge_case_test(n_samples=35, start_id=5001, seed=46) -> pd.DataFrame:
    np.random.seed(seed)
    limit_bal = np.random.choice([5000, 10000, 500000, 1000000], n_samples)
    sex = np.random.choice([1, 2], n_samples)
    # Undocumented category values for EDUCATION (0, 5, 6) and MARRIAGE (0)
    education = np.random.choice([0, 1, 2, 3, 5, 6], n_samples)
    marriage = np.random.choice([0, 1, 2, 3], n_samples)
    age = np.random.choice([18, 19, 79, 85], n_samples)
    
    pay_0 = np.random.choice([-2, 0, 8], n_samples)
    pay_2 = np.random.choice([-2, 0, 7], n_samples)
    pay_3 = np.random.choice([-2, 0], n_samples)
    pay_4 = np.random.choice([-2, 0], n_samples)
    pay_5 = np.random.choice([-2, 0], n_samples)
    pay_6 = np.random.choice([-2, 0], n_samples)
    
    # Negative bill amounts (credit refunds) and zero bill amounts
    bill_amt1 = np.random.choice([-1500.0, 0.0, 450000.0], n_samples)
    bill_amt2 = np.random.choice([-500.0, 0.0, 420000.0], n_samples)
    bill_amt3 = np.zeros(n_samples)
    bill_amt4 = np.zeros(n_samples)
    bill_amt5 = np.zeros(n_samples)
    bill_amt6 = np.zeros(n_samples)
    
    pay_amt1 = np.zeros(n_samples)
    pay_amt2 = np.zeros(n_samples)
    pay_amt3 = np.zeros(n_samples)
    pay_amt4 = np.zeros(n_samples)
    pay_amt5 = np.zeros(n_samples)
    pay_amt6 = np.zeros(n_samples)
    
    default_target = np.random.choice([0, 1], n_samples)
    
    df = pd.DataFrame({
        'ID': range(start_id, start_id + n_samples),
        'customer_name': generate_names(n_samples, seed),
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2,
        'BILL_AMT3': bill_amt3, 'BILL_AMT4': bill_amt4,
        'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2,
        'PAY_AMT3': pay_amt3, 'PAY_AMT4': pay_amt4,
        'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6,
        'default.payment.next.month': default_target
    })
    return df

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_csvs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Target directory verified: {output_dir}")

    generators = [
        ('high_risk_portfolio.csv', create_high_risk_portfolio),
        ('medium_risk_portfolio.csv', create_medium_risk_portfolio),
        ('low_risk_portfolio.csv', create_low_risk_portfolio),
        ('mixed_batch_alpha.csv', create_mixed_batch_alpha),
        ('edge_case_test.csv', create_edge_case_test)
    ]

    for fname, func in generators:
        df = func()
        fpath = os.path.join(output_dir, fname)
        df.to_csv(fpath, index=False)
        print(f"Generated {fname} ({len(df)} rows) -> {fpath}")

    print("\nAll 5 sample CSV files successfully generated in sample_csvs/")

if __name__ == '__main__':
    main()
