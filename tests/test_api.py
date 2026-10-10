import unittest
from fastapi.testclient import TestClient
import sys
import os

# Ensure the api module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.index import app
import api.index

client = TestClient(app)

def mock_load_queue_artifact():
    return [
        {
            "customer_id": 101, 
            "customer_name": "Eleanor Vance",
            "default_probability": 0.95, 
            "risk_tier": "High Risk: Priority Outbound Call", 
            "top_risk_features": ["Recent Payment Delay"],
            "explanation": "Customer has missed payments for 2 consecutive billing cycles."
        },
        {
            "customer_id": 102, 
            "customer_name": "Marcus Chen",
            "default_probability": 0.65, 
            "risk_tier": "Medium Risk: Proactive Limit Review", 
            "top_risk_features": ["High Line Utilization"],
            "explanation": "Recent payment covers less than 10% of statement balance."
        },
        {
            "customer_id": 103, 
            "customer_name": "Sarah Jenkins",
            "default_probability": 0.35, 
            "risk_tier": "Early Warning: Digital SMS Nudge", 
            "top_risk_features": [],
            "explanation": "Early volatility indicators detected in revolving payment ratios."
        },
    ]

api.index.load_queue_artifact = mock_load_queue_artifact

class TestCreditWiseAPI(unittest.TestCase):
    def test_draft_outreach_options(self):
        response = client.post("/api/draft-outreach", json={"customer_id": 101, "action_key": "HARDSHIP_PLAN"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], 101)
        self.assertEqual(data["customer_name"], "Eleanor Vance")
        self.assertEqual(data["status"], "PENDING_APPROVAL")
        self.assertEqual(data["selected_action_key"], "HARDSHIP_PLAN")
        self.assertEqual(len(data["available_actions"]), 4)

    def test_execute_action(self):
        response = client.post("/api/execute-action", json={
            "customer_id": 101,
            "action_key": "HARDSHIP_PLAN",
            "approved_by": "Collections Officer"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "EXECUTED")
        self.assertEqual(data["customer_name"], "Eleanor Vance")

    def test_draft_outreach_not_found(self):
        response = client.post("/api/draft-outreach", json={"customer_id": 999})
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

    def test_draft_outreach_validation_error(self):
        response = client.post("/api/draft-outreach", json={"customer_id": "invalid_id"})
        self.assertEqual(response.status_code, 422)

    def test_draft_outreach_missing_body(self):
        response = client.post("/api/draft-outreach", json={})
        self.assertEqual(response.status_code, 422)

    def test_assess_customer(self):
        response = client.post("/api/assess-customer", json={
            "customer_name": "Test User",
            "limit_bal": 75000,
            "pay_0": 2,
            "bill_amt1": 60000,
            "pay_amt1": 1000
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["assessment"]["customer_name"], "Test User")
        self.assertIn("High Risk", data["assessment"]["risk_tier"])

    def test_negotiate_hardship(self):
        response = client.post("/api/negotiate-hardship", json={
            "customer_id": 101,
            "requested_plan_type": "reduced_monthly",
            "counter_offer_note": "Requesting 50% lower monthly payment"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], 101)
        self.assertEqual(data["status"], "PENDING_APPROVAL")
        self.assertIn("POLICY GUARDRAIL", data["policy_guardrail_applied"])
        self.assertEqual(len(data["payment_schedule"]), 6)

    def test_approve_hardship(self):
        response = client.post("/api/approve-hardship", json={
            "customer_id": 101,
            "decision": "APPROVE",
            "approved_by": "Senior Risk Officer"
        })
    def test_rbac_customer_forbidden(self):
        response = client.get("/api/dashboard-metrics", headers={"X-User-Role": "customer"})
        self.assertEqual(response.status_code, 403)

    def test_customer_negotiate(self):
        response = client.post("/api/customer/negotiate", json={
            "customer_id": 101,
            "requested_plan_type": "freeze_interest"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], 101)
        self.assertIn("0% Zero-Interest", data["approved_plan_title"])

if __name__ == "__main__":
    unittest.main()



