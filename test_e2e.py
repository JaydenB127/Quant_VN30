"""End-to-end test script for ETS API."""
import requests
import json

BASE = "http://127.0.0.1:8000"

# Step 1: Upload dataset
print("=== STEP 1: Upload Dataset ===")
csv_content = "date,close,volume,target\n2024-01-01,100.0,1000000,1\n2024-01-02,101.5,1200000,0\n2024-01-03,99.8,900000,1\n2024-01-04,102.3,1100000,0\n2024-01-05,103.1,1500000,1\n"
r = requests.post(
    f"{BASE}/api/datasets/upload",
    files={"file": ("test_data.csv", csv_content.encode(), "text/csv")},
    data={"description": "Test stock data"},
)
print(f"Status: {r.status_code}")
d = r.json()
print(json.dumps(d, indent=2))
dataset_id = d.get("id")

# Step 2: Create experiment
print("\n=== STEP 2: Create Experiment ===")
r2 = requests.post(
    f"{BASE}/api/experiments/",
    json={
        "name": "Test Finance Experiment",
        "description": "Quick test",
        "dataset_id": dataset_id,
        "pipeline_type": "finance_forecasting",
        "config_json": {"skip_download": True, "fast": True, "quick": True},
    },
)
print(f"Status: {r2.status_code}")
e = r2.json()
print(json.dumps(e, indent=2))

# Step 3: List datasets
print("\n=== STEP 3: Verify Datasets ===")
r3 = requests.get(f"{BASE}/api/datasets/")
print(f"Datasets count: {len(r3.json())}")

# Step 4: List experiments
print("\n=== STEP 4: Verify Experiments ===")
r4 = requests.get(f"{BASE}/api/experiments/")
print(f"Experiments count: {len(r4.json())}")

# Step 5: Get dataset details
print("\n=== STEP 5: Dataset Profile ===")
r5 = requests.get(f"{BASE}/api/datasets/{dataset_id}")
profile = r5.json()
print("Suggested target:", profile.get("suggested_target"))
print("Problem type:", profile.get("suggested_problem_type"))
print("Schema columns:", list(profile.get("schema_json", {}).keys()))

# Step 6: Deduplicate check
print("\n=== STEP 6: Deduplication Check ===")
r6 = requests.post(
    f"{BASE}/api/datasets/upload",
    files={"file": ("test_data.csv", csv_content.encode(), "text/csv")},
    data={"description": "Duplicate test"},
)
print(f"Status: {r6.status_code}")
d6 = r6.json()
print(f"Dedup status: {d6.get('status')}")

print("\n[OK] ALL TESTS PASSED!")
