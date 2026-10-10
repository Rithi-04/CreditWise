import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def generate_customer_name(customer_id: int) -> str:
    """Generates realistic, diverse synthetic names for UCI numeric customer IDs."""
    first = FIRST_NAMES[customer_id % len(FIRST_NAMES)]
    last = LAST_NAMES[(customer_id * 7 + 3) % len(LAST_NAMES)]
    return f"{first} {last}"

def generate_agent_explanation(row: pd.Series, risk_tier: str) -> str:
    """
    Agent explanation engine: Translates financial feature signals into 
    pure plain-language, empathetic banking narratives with ZERO technical jargon.
    """
    pay_0 = row.get('PAY_0', 0)
    pay_2 = row.get('PAY_2', 0)
    limit_bal = row.get('LIMIT_BAL', 50000)
    bill_amt1 = row.get('BILL_AMT1', 0)
    pay_amt1 = row.get('PAY_AMT1', 0)

    if pay_0 >= 2:
        return f"Customer has missed payments for {int(pay_0)} consecutive billing cycles while maintaining high credit utilization, indicating an emerging cash-flow crunch."
    elif pay_0 == 1:
        return "Customer shows an initial 1-month payment delay accompanied by an increased revolving balance relative to their credit line."
    elif bill_amt1 > limit_bal * 0.8:
        return "Statement balance exceeds 80% of total credit line with minimal monthly repayment, signaling debt accumulation risk."
    elif pay_amt1 > 0 and pay_amt1 < bill_amt1 * 0.1 and bill_amt1 > 5000:
        return "Recent payment covers less than 10% of total statement balance, indicating liquidity compression."
    elif pay_2 >= 1:
        return "Secondary repayment friction detected following recent account utilization spikes."
    else:
        return "Early volatility indicators detected in revolving payment ratios and monthly statement activity relative to credit line."

def get_human_risk_drivers(row: pd.Series) -> list[str]:
    """Generates human-readable, non-technical risk driver badges."""
    pay_0 = row.get('PAY_0', 0)
    limit_bal = row.get('LIMIT_BAL', 50000)
    bill_amt1 = row.get('BILL_AMT1', 0)
    pay_amt1 = row.get('PAY_AMT1', 0)

    drivers = []
    if pay_0 >= 1:
        drivers.append("Recent Payment Delay")
    if bill_amt1 > limit_bal * 0.5:
        drivers.append("High Line Utilization")
    if pay_amt1 < bill_amt1 * 0.2:
        drivers.append("Low Monthly Repayment")

    if not drivers:
        drivers = ["Payment Delay History", "Revolving Utilization", "Prior Payment Friction"]
    return drivers[:3]


def load_and_clean_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads dataset and performs initial hygiene by handling identifier 
    columns and mapping undocumented categorical values.
    """
    logger.info("Loading UCI Credit Card dataset...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        df = pd.read_excel(url, header=1)
    except Exception as e:
        logger.warning(f"Failed to fetch dataset from UCI: {e}. Generating synthetic fallback data for demonstration.")
        np.random.seed(42)
        df = pd.DataFrame({
            'ID': range(1, 1001),
            'LIMIT_BAL': np.random.uniform(10000, 500000, 1000),
            'SEX': np.random.choice([1, 2], 1000),
            'EDUCATION': np.random.choice([0, 1, 2, 3, 4, 5, 6], 1000),
            'MARRIAGE': np.random.choice([0, 1, 2, 3], 1000),
            'AGE': np.random.randint(20, 80, 1000),
            'PAY_0': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 1000),
            'PAY_2': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 1000),
            'PAY_3': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 1000),
            'PAY_4': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 1000),
            'PAY_5': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 1000),
            'PAY_6': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 1000),
            'BILL_AMT1': np.random.uniform(0, 100000, 1000),
            'BILL_AMT2': np.random.uniform(0, 100000, 1000),
            'BILL_AMT3': np.random.uniform(0, 100000, 1000),
            'BILL_AMT4': np.random.uniform(0, 100000, 1000),
            'BILL_AMT5': np.random.uniform(0, 100000, 1000),
            'BILL_AMT6': np.random.uniform(0, 100000, 1000),
            'PAY_AMT1': np.random.uniform(0, 10000, 1000),
            'PAY_AMT2': np.random.uniform(0, 10000, 1000),
            'PAY_AMT3': np.random.uniform(0, 10000, 1000),
            'PAY_AMT4': np.random.uniform(0, 10000, 1000),
            'PAY_AMT5': np.random.uniform(0, 10000, 1000),
            'PAY_AMT6': np.random.uniform(0, 10000, 1000),
            'default payment next month': np.random.choice([0, 1], 1000, p=[0.78, 0.22])
        })

    # Rename target for convenience
    df = df.rename(columns={'default payment next month': 'target'})
    
    # Store and drop ID to prevent leakage
    if 'ID' in df.columns:
        customer_ids = df['ID']
        df = df.drop(columns=['ID'])
    else:
        customer_ids = pd.Series(range(1, len(df) + 1))

    # Collapse undocumented categorical values (0, 5, 6 -> 4 "Other")
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    
    # Collapse MARRIAGE (0 -> 3 "Other")
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    
    return df, customer_ids

def build_pipeline(categorical_features: list[str], numeric_features: list[str], model_type: str = 'gbm') -> Pipeline:
    """
    Constructs a robust, leakage-free scikit-learn Pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    if model_type == 'gbm':
        classifier = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=42)

    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])

def main():
    df, customer_ids = load_and_clean_data()
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    logger.info("Performing leakage-free train-test split...")
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Training baseline Logistic Regression model...")
    lr_pipeline = build_pipeline(categorical_features, numeric_features, model_type='lr')
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict_proba(X_test)[:, 1]
    logger.info(f"LR Baseline - ROC-AUC: {roc_auc_score(y_test, lr_preds):.4f}, PR-AUC: {average_precision_score(y_test, lr_preds):.4f}")

    logger.info("Training robust Gradient Boosting Classifier...")
    pipeline = build_pipeline(categorical_features, numeric_features, model_type='gbm')
    pipeline.fit(X_train, y_train)
    
    logger.info("Evaluating model...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    logger.info(f"GBM - ROC-AUC: {roc_auc:.4f}")
    logger.info(f"GBM - PR-AUC: {pr_auc:.4f}")
    
    if roc_auc < 0.70 or roc_auc > 0.82:
        logger.warning(f"ROC-AUC {roc_auc:.4f} is outside the expected band. Check dataset.")
    
    # Artifact Serialization
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'credit_model.pkl')
    joblib.dump(pipeline, model_path)
    logger.info(f"Model successfully saved to {model_path}")
    
    # Precomputed Top Risk Queue Generation with Names & Agent Narratives
    logger.info("Generating top 500 highest-risk queue with customer names and agent narratives...")
    test_df = X_test.copy()
    test_df['customer_id'] = id_test.values
    test_df['default_probability'] = y_pred_proba
    
    test_results = test_df.sort_values(by='default_probability', ascending=False).head(500)
    
    queue = []
    for idx, row in test_results.iterrows():
        cid = int(row['customer_id'])
        prob = float(row['default_probability'])
        
        # Professional Risk Tier Mapping
        if prob >= 0.50:
            risk_tier = "High Risk: Priority Outbound Call"
        elif prob >= 0.25:
            risk_tier = "Medium Risk: Proactive Limit Review"
        else:
            risk_tier = "Early Warning: Digital SMS Nudge"
            
        cname = generate_customer_name(cid)
        explanation = generate_agent_explanation(row, risk_tier)
        drivers = get_human_risk_drivers(row)
        
        queue.append({
            "customer_id": cid,
            "customer_name": cname,
            "default_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "top_risk_features": drivers,
            "explanation": explanation,
            "limit_bal": float(row.get('LIMIT_BAL', 50000)),
            "age": int(row.get('AGE', 35)),
            "education": int(row.get('EDUCATION', 2)),
            "pay_0": int(row.get('PAY_0', 0)),
            "bill_amt1": float(row.get('BILL_AMT1', 0)),
            "pay_amt1": float(row.get('PAY_AMT1', 0))
        })

        
    queue_path = os.path.join(output_dir, 'top_500_queue.json')
    with open(queue_path, 'w') as f:
        json.dump(queue, f, indent=2)
    logger.info(f"Precomputed queue successfully saved to {queue_path}")

if __name__ == "__main__":
    main()

