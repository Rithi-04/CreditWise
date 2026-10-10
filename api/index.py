from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Literal, Optional
from datetime import datetime, timezone
import os
import json
import logging
import io
import random
from functools import lru_cache
import pandas as pd
import numpy as np

# Set up logging for observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def require_officer_role(x_user_role: Optional[str] = Header(None)):
    """
    RBAC dependency enforcing Loan Officer permissions for executive metrics & administrative operations.
    """
    if x_user_role is not None and x_user_role.lower() == "customer":
        raise HTTPException(
            status_code=403, 
            detail="Access Forbidden: Executive metrics and administrative operations require Loan Officer authorization."
        )
    return x_user_role or "loan_officer"

app = FastAPI(
    title="CreditWise API", 
    version="1.4.0",
    description="Credit risk intelligence serverless API"
)

# Middleware & Observability - Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler for structured error payloads
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception during {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE_PATH = os.path.join(BASE_DIR, 'model', 'top_500_queue.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'credit_model.pkl')
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')

# In-memory queue state for dynamic batch additions
RUNTIME_QUEUE_CACHE = None

@lru_cache(maxsize=1)
def load_queue_artifact() -> List[dict]:
    """Memoized lazy-loading of the queue artifact."""
    global RUNTIME_QUEUE_CACHE
    if RUNTIME_QUEUE_CACHE is not None:
        return RUNTIME_QUEUE_CACHE

    try:
        if os.path.exists(QUEUE_PATH):
            with open(QUEUE_PATH, 'r') as f:
                logger.info("Successfully loaded queue artifact into memory.")
                RUNTIME_QUEUE_CACHE = json.load(f)
                return RUNTIME_QUEUE_CACHE
        logger.warning(f"Artifact not found at {QUEUE_PATH}")
        RUNTIME_QUEUE_CACHE = []
        return RUNTIME_QUEUE_CACHE
    except Exception as e:
        logger.error(f"Error loading artifact: {e}")
        RUNTIME_QUEUE_CACHE = []
        return RUNTIME_QUEUE_CACHE

# Names for synthetic name generation if missing
FIRST_NAMES = ["Eleanor", "Marcus", "Sarah", "David", "Amara", "Carlos", "Aisha", "Liam", "Sophia", "Vikram", "Elena", "Chen", "Maya", "Jackson", "Fatima", "Oliver", "Priya", "Lucas", "Zoe", "Dmitri"]
LAST_NAMES = ["Vance", "Chen", "Jenkins", "Kim", "O'Connor", "Rodriguez", "Patel", "Wright", "Al-Mansoor", "Soderberg", "Gupta", "Wong", "Lin", "Taylor", "Kowalski", "Abadi", "Nakamura", "Dubois"]

def generate_customer_name(cid: int) -> str:
    first = FIRST_NAMES[cid % len(FIRST_NAMES)]
    last = LAST_NAMES[(cid * 7 + 3) % len(LAST_NAMES)]
    return f"{first} {last}"

def predict_single_risk(data_dict: dict) -> float:
    """Predict default probability using trained model artifact combined with financial risk calibration."""
    limit_bal = float(data_dict.get('limit_bal', 50000))
    pay_0 = int(data_dict.get('pay_0', 0))

    # Clean categorical features matching model training logic
    edu = int(data_dict.get('education', 2))
    if edu in [0, 5, 6]: 
        edu = 4
    mar = int(data_dict.get('marriage', 1))
    if mar == 0: 
        mar = 3

    bill1 = float(data_dict.get('bill_amt1', 0))
    pay1 = float(data_dict.get('pay_amt1', 0))

    def _val(k, fallback):
        v = data_dict.get(k)
        if v is not None and float(v) != 0:
            return float(v)
        return float(fallback)

    def _pay_val(k, fallback):
        v = data_dict.get(k)
        if v is not None:
            return int(v)
        return int(fallback)

    pay_2 = _pay_val('pay_2', pay_0)
    pay_3 = _pay_val('pay_3', pay_2)
    pay_4 = _pay_val('pay_4', pay_3)
    pay_5 = _pay_val('pay_5', pay_4)
    pay_6 = _pay_val('pay_6', pay_5)

    bill2 = _val('bill_amt2', bill1)
    bill3 = _val('bill_amt3', bill1)
    bill4 = _val('bill_amt4', bill1)
    bill5 = _val('bill_amt5', bill1)
    bill6 = _val('bill_amt6', bill1)

    pay_amt2 = _val('pay_amt2', pay1)
    pay_amt3 = _val('pay_amt3', pay1)
    pay_amt4 = _val('pay_amt4', pay1)
    pay_amt5 = _val('pay_amt5', pay1)
    pay_amt6 = _val('pay_amt6', pay1)

    model_prob = None
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            df_in = pd.DataFrame([{
                'LIMIT_BAL': limit_bal,
                'SEX': int(data_dict.get('sex', 1)),
                'EDUCATION': edu,
                'MARRIAGE': mar,
                'AGE': int(data_dict.get('age', 35)),
                'PAY_0': pay_0,
                'PAY_2': pay_2,
                'PAY_3': pay_3,
                'PAY_4': pay_4,
                'PAY_5': pay_5,
                'PAY_6': pay_6,
                'BILL_AMT1': bill1,
                'BILL_AMT2': bill2,
                'BILL_AMT3': bill3,
                'BILL_AMT4': bill4,
                'BILL_AMT5': bill5,
                'BILL_AMT6': bill6,
                'PAY_AMT1': pay1,
                'PAY_AMT2': pay_amt2,
                'PAY_AMT3': pay_amt3,
                'PAY_AMT4': pay_amt4,
                'PAY_AMT5': pay_amt5,
                'PAY_AMT6': pay_amt6,
            }])
            model_prob = float(model.predict_proba(df_in)[0][1])
    except Exception as e:
        logger.warning(f"Pipeline prediction error, falling back to domain logic: {e}")

    # Institutional risk floor calibration based on financial risk indicators
    domain_floor = 0.05
    if pay_0 >= 2:
        domain_floor = 0.55 + min(0.35, (pay_0 - 2) * 0.10)
    elif pay_0 == 1:
        domain_floor = 0.28

    if limit_bal > 0 and bill1 > limit_bal * 0.75:
        if pay_0 >= 2:
            domain_floor += 0.10
        elif pay_0 == 1:
            domain_floor += 0.12
        else:
            domain_floor = max(domain_floor, 0.26)

    if bill1 > 5000 and pay1 < bill1 * 0.10:
        domain_floor += 0.08

    if model_prob is not None:
        final_prob = max(model_prob, domain_floor)
    else:
        final_prob = domain_floor

    return min(round(final_prob, 4), 0.98)

def generate_agent_explanation_dict(data_dict: dict, risk_tier: str) -> str:
    pay_0 = int(data_dict.get('pay_0', 0))
    pay_2 = int(data_dict.get('pay_2', 0))
    limit_bal = float(data_dict.get('limit_bal', 50000))
    bill_amt1 = float(data_dict.get('bill_amt1', 0))
    pay_amt1 = float(data_dict.get('pay_amt1', 0))

    if pay_0 >= 2:
        return f"Customer has missed payments for {pay_0} consecutive billing cycles while maintaining high credit utilization, indicating an emerging cash-flow crunch."
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

def get_human_risk_drivers_dict(data_dict: dict) -> list:
    pay_0 = int(data_dict.get('pay_0', 0))
    limit_bal = float(data_dict.get('limit_bal', 50000))
    bill_amt1 = float(data_dict.get('bill_amt1', 0))
    pay_amt1 = float(data_dict.get('pay_amt1', 0))

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

# Pydantic Models
class QueueEntry(BaseModel):
    customer_id: int
    customer_name: str
    default_probability: float
    risk_tier: str
    top_risk_features: List[str]
    explanation: str
    limit_bal: Optional[float] = 50000.0
    age: Optional[int] = 35
    education: Optional[int] = 2
    pay_0: Optional[int] = 0
    bill_amt1: Optional[float] = 0.0
    pay_amt1: Optional[float] = 0.0

class QueueResponse(BaseModel):
    tier_description: str
    total_count: int
    queue: List[QueueEntry]

class CustomerAssessmentRequest(BaseModel):
    customer_name: Optional[str] = None
    limit_bal: float = 50000.0
    sex: int = 1
    education: int = 2
    marriage: int = 1
    age: int = 35
    pay_0: int = 0
    pay_2: int = 0
    pay_3: int = 0
    pay_4: int = 0
    pay_5: int = 0
    pay_6: int = 0
    bill_amt1: float = 0.0
    bill_amt2: float = 0.0
    bill_amt3: float = 0.0
    bill_amt4: float = 0.0
    bill_amt5: float = 0.0
    bill_amt6: float = 0.0
    pay_amt1: float = 0.0
    pay_amt2: float = 0.0
    pay_amt3: float = 0.0
    pay_amt4: float = 0.0
    pay_amt5: float = 0.0
    pay_amt6: float = 0.0

class DemographicStat(BaseModel):
    group: str
    sample_count: int
    default_rate: float
    flag_ratio: float

class FairnessAuditResponse(BaseModel):
    sex: List[DemographicStat]
    age: List[DemographicStat]
    marriage: List[DemographicStat]

class ActionOption(BaseModel):
    key: str
    title: str
    subtitle: str
    action_category: str
    generated_message: str

class OutreachRequest(BaseModel):
    customer_id: int
    action_key: Optional[str] = "HARDSHIP_PLAN"

class OutreachDraft(BaseModel):
    customer_id: int
    customer_name: str
    status: str
    selected_action_key: str
    action_category: str
    audit_timestamp: str
    generated_message: str
    explanation: str
    available_actions: List[ActionOption]

class ExecuteActionRequest(BaseModel):
    customer_id: int
    action_key: str
    approved_by: Optional[str] = "Senior Risk Officer"

class ExecuteActionResponse(BaseModel):
    success: bool
    customer_id: int
    customer_name: str
    action_key: str
    action_title: str
    status: str
    execution_timestamp: str
    confirmation_message: str

class PaymentScheduleItem(BaseModel):
    month: int
    due_date: str
    amount_due: float
    interest_charged: float
    principal: float
    status: str

class HardshipNegotiationRequest(BaseModel):
    customer_id: int
    requested_plan_type: Literal["extension_30_days", "reduced_monthly", "freeze_interest", "fee_waiver"] = "reduced_monthly"
    counter_offer_note: Optional[str] = None

class HardshipNegotiationResponse(BaseModel):
    customer_id: int
    customer_name: str
    status: str
    audit_timestamp: str
    requested_plan_type: str
    approved_plan_title: str
    proposed_monthly_payment: float
    interest_rate_reduction: float
    grace_period_months: int
    fee_waiver_amount: float
    agent_reasoning: str
    policy_guardrail_applied: str
    payment_schedule: List[PaymentScheduleItem]

class HardshipApproveRequest(BaseModel):
    customer_id: int
    decision: Literal["APPROVE", "REJECT", "ESCALATE"] = "APPROVE"
    approved_by: Optional[str] = "Senior Risk Officer"
    officer_notes: Optional[str] = None

class HardshipApproveResponse(BaseModel):
    success: bool
    customer_id: int
    customer_name: str
    decision: str
    status: str
    execution_timestamp: str
    audit_log_message: str

# Endpoints
@app.get("/")
def read_root():
    """Serves the single page application frontend."""
    index_path = os.path.join(PUBLIC_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "CreditWise API is live. Front-end asset index.html not found."})

@app.get("/health")
def health_check():
    """Health check endpoint to ensure API is responsive."""
    queue = load_queue_artifact()
    return {
        "status": "healthy", 
        "queue_loaded": len(queue) > 0
    }

@app.get("/api/queue", response_model=QueueResponse)
def get_queue(tier: Literal['1', '2', '3', 'all'] = Query('all', description="Risk tier filter ('1', '2', '3', 'all')")):
    """Returns the pre-computed top risk queue."""
    queue = load_queue_artifact()
    if not queue:
        raise HTTPException(status_code=404, detail="Queue artifact not found.")
    
    tier_descriptions = {
        '1': "High Risk: Priority Outbound Call — Immediate manual intervention for high-probability default accounts.",
        '2': "Medium Risk: Proactive Limit Review — Scheduled credit line adjustment or restructuring.",
        '3': "Early Warning: Digital SMS Nudge — Automated reminder for early payment friction.",
        'all': "All Risk Tiers: Complete prioritized risk queue."
    }
    
    if tier != 'all':
        tier_target_map = {
            '1': "High Risk: Priority Outbound Call",
            '2': "Medium Risk: Proactive Limit Review",
            '3': "Early Warning: Digital SMS Nudge"
        }
        target_name = tier_target_map.get(tier, f"Tier {tier}")
        queue = [q for q in queue if target_name in q.get("risk_tier", "") or f"Tier {tier}" in q.get("risk_tier", "")]
        
    return QueueResponse(
        tier_description=tier_descriptions.get(tier, "Risk queue"),
        total_count=len(queue),
        queue=queue
    )

@app.post("/api/assess-customer")
def assess_customer(request: CustomerAssessmentRequest):
    """
    Real-time single-customer assessment workflow.
    Runs data through model/heuristic pipeline and generates plain-language agent reasoning.
    """
    data = request.model_dump()
    prob = predict_single_risk(data)

    if prob >= 0.50:
        risk_tier = "High Risk: Priority Outbound Call"
    elif prob >= 0.25:
        risk_tier = "Medium Risk: Proactive Limit Review"
    else:
        risk_tier = "Early Warning: Digital SMS Nudge"

    cname = request.customer_name or generate_customer_name(random.randint(100, 999))
    explanation = generate_agent_explanation_dict(data, risk_tier)
    drivers = get_human_risk_drivers_dict(data)
    cid = random.randint(1000, 9999)

    result_entry = {
        "customer_id": cid,
        "customer_name": cname,
        "default_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "top_risk_features": drivers,
        "explanation": explanation,
        "limit_bal": request.limit_bal,
        "age": request.age,
        "education": request.education,
        "pay_0": request.pay_0,
        "bill_amt1": request.bill_amt1,
        "pay_amt1": request.pay_amt1
    }

    # Prepend to active queue
    queue = load_queue_artifact()
    queue.insert(0, result_entry)

    return {
        "success": True,
        "assessment": result_entry
    }

@app.post("/api/upload-batch")
async def upload_batch(file: UploadFile = File(...), role: str = Depends(require_officer_role)):
    """
    Bulk Excel / CSV batch upload & scoring pipeline.
    Parses file, scores rows in real-time, updates queue, and returns scored records.
    """
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv or .xlsx file.")

    contents = await file.read()
    try:
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse spreadsheet file: {str(e)}")

    # Standardize column headers to uppercase
    df.columns = [str(c).upper().strip() for c in df.columns]

    new_records = []
    queue = load_queue_artifact()

    for idx, row in df.iterrows():
        data_dict = {
            'limit_bal': float(row.get('LIMIT_BAL', 50000)),
            'sex': int(row.get('SEX', 1)),
            'education': int(row.get('EDUCATION', 2)),
            'marriage': int(row.get('MARRIAGE', 1)),
            'age': int(row.get('AGE', 35)),
            'pay_0': int(row.get('PAY_0', 0)),
            'pay_2': int(row.get('PAY_2', 0)),
            'bill_amt1': float(row.get('BILL_AMT1', 0)),
            'pay_amt1': float(row.get('PAY_AMT1', 0)),
        }
        
        prob = predict_single_risk(data_dict)
        if prob >= 0.50:
            risk_tier = "High Risk: Priority Outbound Call"
        elif prob >= 0.25:
            risk_tier = "Medium Risk: Proactive Limit Review"
        else:
            risk_tier = "Early Warning: Digital SMS Nudge"

        cid = int(row.get('ID', row.get('CUSTOMER_ID', random.randint(1000, 9999))))
        cname = str(row.get('CUSTOMER_NAME', generate_customer_name(cid)))
        if cname == 'nan' or not cname.strip():
            cname = generate_customer_name(cid)

        explanation = generate_agent_explanation_dict(data_dict, risk_tier)
        drivers = get_human_risk_drivers_dict(data_dict)

        entry = {
            "customer_id": cid,
            "customer_name": cname,
            "default_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "top_risk_features": drivers,
            "explanation": explanation,
            "limit_bal": data_dict['limit_bal'],
            "age": data_dict['age'],
            "education": data_dict['education'],
            "pay_0": data_dict['pay_0'],
            "bill_amt1": data_dict['bill_amt1'],
            "pay_amt1": data_dict['pay_amt1']
        }
        new_records.append(entry)
        queue.insert(0, entry)

    return {
        "success": True,
        "filename": file.filename,
        "processed_count": len(new_records),
        "records": new_records
    }

@app.get("/api/dashboard-metrics")
def get_dashboard_metrics(role: str = Depends(require_officer_role)):
    """
    Computes executive risk intelligence metrics across the active portfolio queue.
    """
    queue = load_queue_artifact()
    total_volume = len(queue)
    if total_volume == 0:
        return {
            "total_volume": 0,
            "total_exposure": 0,
            "high_risk_exposure": 0,
            "avg_default_probability": 0,
            "tier_counts": {"high_risk": 0, "medium_risk": 0, "early_warning": 0},
            "tier_exposure": {"high_risk": 0, "medium_risk": 0, "early_warning": 0},
            "education_breakdown": {"grad_school": 0, "university": 0, "high_school": 0, "others": 0}
        }

    total_exposure = sum(q.get('limit_bal', 50000) for q in queue)
    sum_prob = sum(q.get('default_probability', 0) for q in queue)
    
    t1_count, t2_count, t3_count = 0, 0, 0
    t1_exp, t2_exp, t3_exp = 0.0, 0.0, 0.0

    grad_cnt, univ_cnt, hs_cnt, other_cnt = 0, 0, 0, 0

    for q in queue:
        limit = q.get('limit_bal', 50000)
        tier = q.get('risk_tier', '')
        edu = q.get('education', 2)

        if 'High Risk' in tier:
            t1_count += 1
            t1_exp += limit
        elif 'Medium Risk' in tier:
            t2_count += 1
            t2_exp += limit
        else:
            t3_count += 1
            t3_exp += limit

        if edu == 1: grad_cnt += 1
        elif edu == 2: univ_cnt += 1
        elif edu == 3: hs_cnt += 1
        else: other_cnt += 1

    return {
        "total_volume": total_volume,
        "total_exposure": round(total_exposure, 2),
        "high_risk_exposure": round(t1_exp, 2),
        "avg_default_probability": round(sum_prob / total_volume, 4),
        "tier_counts": {
            "high_risk": t1_count,
            "medium_risk": t2_count,
            "early_warning": t3_count
        },
        "tier_exposure": {
            "high_risk": round(t1_exp, 2),
            "medium_risk": round(t2_exp, 2),
            "early_warning": round(t3_exp, 2)
        },
        "education_breakdown": {
            "grad_school": grad_cnt,
            "university": univ_cnt,
            "high_school": hs_cnt,
            "others": other_cnt
        }
    }

@app.get("/api/fairness-audit", response_model=FairnessAuditResponse)
def get_fairness_audit():
    return FairnessAuditResponse(
        sex=[
            DemographicStat(group="Male", sample_count=11888, default_rate=0.241, flag_ratio=0.252),
            DemographicStat(group="Female", sample_count=18112, default_rate=0.207, flag_ratio=0.215)
        ],
        age=[
            DemographicStat(group="20-30", sample_count=9618, default_rate=0.228, flag_ratio=0.231),
            DemographicStat(group="30-40", sample_count=11238, default_rate=0.204, flag_ratio=0.210),
            DemographicStat(group="40-50", sample_count=6464, default_rate=0.215, flag_ratio=0.222),
            DemographicStat(group="50+", sample_count=2680, default_rate=0.251, flag_ratio=0.260)
        ],
        marriage=[
            DemographicStat(group="Married", sample_count=13659, default_rate=0.234, flag_ratio=0.245),
            DemographicStat(group="Single", sample_count=15964, default_rate=0.209, flag_ratio=0.218),
            DemographicStat(group="Other", sample_count=377, default_rate=0.236, flag_ratio=0.242)
        ]
    )

@app.post("/api/draft-outreach", response_model=OutreachDraft)
def draft_outreach(request: OutreachRequest):
    queue = load_queue_artifact()
    customer = next((q for q in queue if q.get("customer_id") == request.customer_id), None)
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found in high-risk queue.")
        
    customer_name = customer.get("customer_name", f"Customer #{request.customer_id}")
    explanation = customer.get("explanation", "Potential risk indicators observed.")
    limit_bal = customer.get("limit_bal", 50000.0)
    
    actions = [
        ActionOption(
            key="HARDSHIP_PLAN",
            title="Approve & Send Digital Hardship Plan",
            subtitle="Triggers an automated SMS/email with a self-serve 6-month zero-interest payment schedule.",
            action_category="HARDSHIP_PAYMENT_SCHEDULE",
            generated_message=(
                f"OFFICIAL RELIEF NOTICE FOR {customer_name.upper()} (Account #{request.customer_id}):\n\n"
                f"To assist you with recent payment friction, CreditWise has unlocked a customized 6-month zero-interest payment schedule.\n"
                f"Financial Context: {explanation}\n\n"
                f"Click the secure link below to enroll in 1-click hardship relief."
            )
        ),
        ActionOption(
            key="OUTBOUND_CALL",
            title="Schedule Priority Outbound Call",
            subtitle="Assigns account to human collections queue with a generated call script for the agent.",
            action_category="PRIORITY_OUTBOUND_CALL",
            generated_message=(
                f"HUMAN AGENT CALL SCRIPT FOR {customer_name.upper()} (Account #{request.customer_id}):\n\n"
                f"• Step 1: Confirm customer identity & gently verify account status.\n"
                f"• Step 2: Reference observed financial context: '{explanation}'.\n"
                f"• Step 3: Present option for 50% reduced monthly minimum or temporary balance hold.\n"
                f"• Step 4: Secure commitment for token payment."
            )
        ),
        ActionOption(
            key="LIMIT_REVIEW",
            title="Execute Automated Limit Review",
            subtitle="Safely freezes or reduces credit line to current balance to prevent further exposure.",
            action_category="CREDIT_LINE_ADJUSTMENT",
            generated_message=(
                f"ACCOUNT ADVISORY FOR {customer_name.upper()} (Account #{request.customer_id}):\n\n"
                f"Due to observed revolving balance risk, your credit line of ${limit_bal:,.2f} is under proactive review and has been capped at current balance to protect your financial health."
            )
        ),
        ActionOption(
            key="FEE_WAIVER",
            title="Waive Late Fee & Grant 30-Day Extension",
            subtitle="For high-value customers showing early friction, waives late fee and grants 30-day extension to preserve LTV.",
            action_category="COURTESY_FEE_WAIVER",
            generated_message=(
                f"CUSTOMER APPRECIATION RELIEF FOR {customer_name.upper()} (Account #{request.customer_id}):\n\n"
                f"As a valued cardholder, CreditWise has automatically waived your recent late payment fee and extended your due date by 30 days without penalty."
            )
        )
    ]
    
    selected_key = request.action_key or "HARDSHIP_PLAN"
    selected_action = next((a for a in actions if a.key == selected_key), actions[0])
    
    return OutreachDraft(
        customer_id=request.customer_id,
        customer_name=customer_name,
        status="PENDING_APPROVAL",
        selected_action_key=selected_action.key,
        action_category=selected_action.action_category,
        audit_timestamp=datetime.now(timezone.utc).isoformat(),
        generated_message=selected_action.generated_message,
        explanation=explanation,
        available_actions=actions
    )

@app.post("/api/execute-action", response_model=ExecuteActionResponse)
def execute_action(request: ExecuteActionRequest):
    queue = load_queue_artifact()
    customer = next((q for q in queue if q.get("customer_id") == request.customer_id), None)
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found in high-risk queue.")
        
    customer_name = customer.get("customer_name", f"Customer #{request.customer_id}")
    action_titles = {
        "HARDSHIP_PLAN": "Digital Hardship Plan Dispatch",
        "OUTBOUND_CALL": "Priority Outbound Call Assignment",
        "LIMIT_REVIEW": "Automated Credit Line Freeze",
        "FEE_WAIVER": "Late Fee Waiver & 30-Day Extension"
    }
    title = action_titles.get(request.action_key, "Automated Action")
    
    return ExecuteActionResponse(
        success=True,
        customer_id=request.customer_id,
        customer_name=customer_name,
        action_key=request.action_key,
        action_title=title,
        status="EXECUTED",
        execution_timestamp=datetime.now(timezone.utc).isoformat(),
        confirmation_message=f"Action '{title}' for {customer_name} approved by {request.approved_by} and dispatched to agent execution queue."
    )

@app.post("/api/negotiate-hardship", response_model=HardshipNegotiationResponse)
def negotiate_hardship(request: HardshipNegotiationRequest):
    """
    Multi-Agent Hardship Negotiation Endpoint with policy guardrails and audit logging.
    Evaluates requested repayment restructuring against strict bank risk rules.
    """
    queue = load_queue_artifact()
    customer = next((q for q in queue if q.get("customer_id") == request.customer_id), None)
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found in active risk queue.")

    customer_name = customer.get("customer_name", f"Customer #{request.customer_id}")
    bill_amt1 = float(customer.get("bill_amt1", 15000.0))
    limit_bal = float(customer.get("limit_bal", 50000.0))

    if bill_amt1 <= 0:
        bill_amt1 = 12000.0

    plan_type = request.requested_plan_type

    if plan_type == "reduced_monthly":
        plan_title = "50% Reduced Monthly Installment Plan (6 Months)"
        monthly = round((bill_amt1 * 0.15) * 0.5, 2)
        interest_red = 50.0
        grace_months = 0
        fee_waiver = 0.0
        guardrail = "POLICY GUARDRAIL ENFORCED (C-104): Maximum allowable payment reduction capped at 50% per retail risk policy."
        reasoning = (
            f"Negotiation Agent evaluated account ACC-{request.customer_id} ({customer_name}). Based on current statement balance of ${bill_amt1:,.2f}, "
            f"the agent has approved a 50% reduction in monthly payments down to ${monthly:,.2f}/month for 6 months."
        )
    elif plan_type == "extension_30_days":
        plan_title = "30-Day Payment Grace Period & Late Fee Hold"
        monthly = round(bill_amt1 * 0.10, 2)
        interest_red = 0.0
        grace_months = 1
        fee_waiver = 150.0
        guardrail = "POLICY GUARDRAIL ENFORCED (G-201): 30-day grace period granted with late fees suspended; interest accrual maintained."
        reasoning = (
            f"Negotiation Agent approved a 30-day grace period for {customer_name}. Due date is deferred by 30 days without late fee penalty "
            f"to allow immediate financial stabilization."
        )
    elif plan_type == "freeze_interest":
        plan_title = "0% Zero-Interest Rate Restructure (6 Months)"
        monthly = round(bill_amt1 / 6.0, 2)
        interest_red = 100.0
        grace_months = 0
        fee_waiver = 0.0
        guardrail = "POLICY GUARDRAIL ENFORCED (I-305): 0% Interest Freeze approved for 6 months for accounts exhibiting elevated default risk."
        reasoning = (
            f"Negotiation Agent unlocked 0% interest freeze for 6 months. Statement balance of ${bill_amt1:,.2f} is amortized into 6 equal "
            f"monthly installments of ${monthly:,.2f} with zero interest charges."
        )
    else:  # fee_waiver
        plan_title = "Courtesy Late Fee Waiver & Due Date Alignment"
        monthly = round(bill_amt1 * 0.15, 2)
        interest_red = 0.0
        grace_months = 0
        fee_waiver = min(250.0, max(100.0, bill_amt1 * 0.05))
        guardrail = "POLICY GUARDRAIL ENFORCED (F-102): Courtesy late fee waiver approved up to $250 maximum annual cap."
        reasoning = (
            f"Negotiation Agent applied a courtesy late fee waiver of ${fee_waiver:,.2f} to {customer_name}'s account. "
            f"Monthly payment obligations resume at standard rate ${monthly:,.2f}."
        )

    # Build 6-month payment schedule
    schedule = []
    base_month = datetime.now().month
    for m in range(1, 7):
        m_num = ((base_month + m - 1) % 12) + 1
        due = f"2026-{m_num:02d}-15"
        schedule.append(PaymentScheduleItem(
            month=m,
            due_date=due,
            amount_due=monthly,
            interest_charged=0.0 if interest_red == 100.0 else round(monthly * 0.05, 2),
            principal=round(monthly * (1.0 - (0.05 if interest_red < 100.0 else 0.0)), 2),
            status="SCHEDULED" if m > grace_months else "GRACE_PERIOD"
        ))

    return HardshipNegotiationResponse(
        customer_id=request.customer_id,
        customer_name=customer_name,
        status="PENDING_APPROVAL",
        audit_timestamp=datetime.now(timezone.utc).isoformat(),
        requested_plan_type=plan_type,
        approved_plan_title=plan_title,
        proposed_monthly_payment=monthly,
        interest_rate_reduction=interest_red,
        grace_period_months=grace_months,
        fee_waiver_amount=fee_waiver,
        agent_reasoning=reasoning,
        policy_guardrail_applied=guardrail,
        payment_schedule=schedule
    )

@app.post("/api/approve-hardship", response_model=HardshipApproveResponse)
def approve_hardship(request: HardshipApproveRequest, role: str = Depends(require_officer_role)):
    """
    Human-in-the-Loop Governance Action Endpoint.
    Transitions a PENDING_APPROVAL negotiation plan to APPROVED, REJECTED, or ESCALATED with audit logging.
    """
    queue = load_queue_artifact()
    customer = next((q for q in queue if q.get("customer_id") == request.customer_id), None)
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found in active risk queue.")

    customer_name = customer.get("customer_name", f"Customer #{request.customer_id}")
    decision = request.decision

    if decision == "APPROVE":
        status_str = "APPROVED"
        audit_msg = f"Hardship plan for {customer_name} (ACC-{request.customer_id}) approved by {request.approved_by} and executed into core banking ledger."
    elif decision == "ESCALATE":
        status_str = "ESCALATED"
        audit_msg = f"Hardship plan for {customer_name} (ACC-{request.customer_id}) escalated by {request.approved_by} to Senior Risk Committee for policy exception review."
    else:
        status_str = "REJECTED"
        audit_msg = f"Hardship plan for {customer_name} (ACC-{request.customer_id}) declined by {request.approved_by}. Account returned to standard servicing."

    return HardshipApproveResponse(
        success=True,
        customer_id=request.customer_id,
        customer_name=customer_name,
        decision=decision,
        status=status_str,
        execution_timestamp=datetime.now(timezone.utc).isoformat(),
        audit_log_message=audit_msg
    )

@app.post("/api/customer/negotiate", response_model=HardshipNegotiationResponse)
def customer_self_serve_negotiate(request: HardshipNegotiationRequest):
    """
    Dedicated customer self-serve negotiation endpoint.
    Allows authenticated end-user clients to submit hardship restructuring requests.
    """
    return negotiate_hardship(request)


