# -*- coding: utf-8 -*-
"""
End-to-end test script to verify deletion features for Datasets, Runs, and Experiments.
"""
import requests
import json
import os
import time

BASE = "http://127.0.0.1:8000"

def test_deletion_flow():
    print("=== STARTING DELETION E2E VERIFICATION ===")
    
    # 1. Ingest a dummy dataset
    print("\n[Step 1] Uploading dummy dataset...")
    csv_content = "date,close,volume,target\n2026-05-25,50000.0,100,1\n2026-05-26,51000.0,120,0\n"
    r_upload = requests.post(
        f"{BASE}/api/datasets/upload",
        files={"file": ("to_be_deleted.csv", csv_content.encode(), "text/csv")},
        data={"description": "Dataset to test deletion flow"}
    )
    assert r_upload.status_code == 200, f"Upload failed: {r_upload.text}"
    dataset_data = r_upload.json()
    dataset_id = dataset_data["id"]
    print(f"Dataset uploaded successfully with ID: {dataset_id}")
    
    # Verify dataset exists in the list
    r_list = requests.get(f"{BASE}/api/datasets/")
    datasets = r_list.json()
    assert any(d["id"] == dataset_id for d in datasets), "Dataset not found in list"
    print("Dataset confirmed present in DB registry.")
    
    # 2. Verify file is written on disk
    dataset_hash = dataset_data["dataset_hash"]
    expected_file_path = os.path.join("outputs", "datasets", f"{dataset_hash}.csv")
    assert os.path.exists(expected_file_path), f"Dataset file not found on disk at {expected_file_path}"
    print(f"Dataset file confirmed on disk at {expected_file_path}")
    
    # 3. Create a test experiment using this dataset
    print("\n[Step 2] Creating experiment associated with the dataset...")
    r_exp = requests.post(
        f"{BASE}/api/experiments/",
        json={
            "name": "Deletion Test Experiment",
            "description": "Experiment created solely for deletion testing",
            "dataset_id": dataset_id,
            "pipeline_type": "finance_forecasting",
            "config_json": {"skip_download": True, "fast": True, "quick": True}
        }
    )
    assert r_exp.status_code == 200, f"Experiment creation failed: {r_exp.text}"
    exp_data = r_exp.json()
    exp_id = exp_data["id"]
    print(f"Experiment created successfully with ID: {exp_id}")
    
    # 4. Try to delete the dataset now (should fail with 400 because it's referenced!)
    print("\n[Step 3] Trying to delete dataset referenced by experiment (safety check)...")
    r_del_ds_fail = requests.delete(f"{BASE}/api/datasets/{dataset_id}")
    print(f"Response status code: {r_del_ds_fail.status_code}")
    print(f"Response detail: {r_del_ds_fail.text}")
    assert r_del_ds_fail.status_code == 400, "Safety check failed: dataset was deleted despite experiment reference!"
    print("Safety check passed! Dataset deletion blocked successfully as expected.")
    
    # 5. Trigger a run for the experiment
    print("\n[Step 4] Triggering a new run for the experiment...")
    r_run = requests.post(f"{BASE}/api/experiments/{exp_id}/run")
    assert r_run.status_code == 200, f"Run trigger failed: {r_run.text}"
    run_data = r_run.json()
    run_id = run_data["run_id"]
    print(f"Run triggered successfully with ID: {run_id}")
    
    # Let's wait a couple of seconds for the background run to start/progress
    print("Waiting for pipeline to initialize and write records/files...")
    time.sleep(3)
    
    # Check run details
    r_run_details = requests.get(f"{BASE}/api/runs/{run_id}")
    assert r_run_details.status_code == 200, f"Failed to get run details: {r_run_details.text}"
    run_details = r_run_details.json()
    print(f"Run current status: {run_details['status']}")
    
    # Check if run has any logged parameters or steps
    r_params = requests.get(f"{BASE}/api/runs/{run_id}/parameters")
    params = r_params.json()
    print(f"Run has {len(params)} parameter(s) logged.")
    
    r_steps = requests.get(f"{BASE}/api/runs/{run_id}/steps")
    steps = r_steps.json()
    print(f"Run has {len(steps)} pipeline step(s) logged.")
    
    # Wait for the run to complete or fail so reports/artifacts are written
    print("Waiting for run completion...")
    for _ in range(15):
        time.sleep(2)
        r_run_details = requests.get(f"{BASE}/api/runs/{run_id}")
        run_details = r_run_details.json()
        status = run_details["status"]
        print(f"Checking run status... {status}")
        if status in ("completed", "failed"):
            break
            
    # Verify if artifacts are registered
    r_arts = requests.get(f"{BASE}/api/runs/{run_id}/artifacts")
    artifacts = r_arts.json()
    print(f"Run has {len(artifacts)} registered artifact(s).")
    
    # Verify artifact files exist on disk if run completed
    artifact_dirs = set()
    artifact_files = []
    for art in artifacts:
        storage_key = art["storage_key"]
        artifact_files.append(storage_key)
        if os.path.exists(storage_key):
            print(f"Artifact file exists: {storage_key}")
        if art.get("metadata_json") and isinstance(art["metadata_json"], dict):
            run_dir = art["metadata_json"].get("run_dir")
            if run_dir:
                artifact_dirs.add(run_dir)
                
    # 6. Delete the run!
    print("\n[Step 5] Deleting the run and its files...")
    r_del_run = requests.delete(f"{BASE}/api/runs/{run_id}")
    assert r_del_run.status_code == 200, f"Delete run failed: {r_del_run.text}"
    print(f"Response: {r_del_run.json()}")
    
    # Verify run DB record is gone
    r_run_details_check = requests.get(f"{BASE}/api/runs/{run_id}")
    assert r_run_details_check.status_code == 404, "Run DB record still exists!"
    print("Run DB record successfully deleted.")
    
    # Verify related tables are cleared (metrics, params, steps, artifacts)
    r_params_check = requests.get(f"{BASE}/api/runs/{run_id}/parameters")
    assert r_params_check.status_code == 200 and len(r_params_check.json()) == 0, "Parameters not cleared!"
    r_steps_check = requests.get(f"{BASE}/api/runs/{run_id}/steps")
    assert r_steps_check.status_code == 200 and len(r_steps_check.json()) == 0, "Steps not cleared!"
    r_arts_check = requests.get(f"{BASE}/api/runs/{run_id}/artifacts")
    assert r_arts_check.status_code == 200 and len(r_arts_check.json()) == 0, "Artifacts not cleared!"
    print("All run sub-tables cleared successfully.")
    
    # Verify artifact files are deleted from disk
    for file_path in artifact_files:
        assert not os.path.exists(file_path), f"Artifact file was not deleted: {file_path}"
    print("All artifact files successfully removed from disk.")
    
    # Verify artifact directories are deleted from disk
    for run_dir in artifact_dirs:
        assert not os.path.exists(run_dir), f"Run directory was not deleted: {run_dir}"
    print("All snapshot run directories successfully removed from disk.")
    
    # 7. Delete the experiment first to release reference on dataset
    print("\n[Step 6] Deleting experiment and verifying cascade deletion...")
    r_del_exp = requests.delete(f"{BASE}/api/experiments/{exp_id}")
    assert r_del_exp.status_code == 200, f"Experiment deletion failed: {r_del_exp.text}"
    print(f"Response: {r_del_exp.json()}")
    
    # Verify experiment DB record is gone
    r_exp_check = requests.get(f"{BASE}/api/experiments/")
    experiments = r_exp_check.json()
    assert not any(e["id"] == exp_id for e in experiments), "Experiment still found in list!"
    print("Experiment successfully deleted.")
    
    # 8. Delete the dataset (should succeed now!)
    print("\n[Step 7] Deleting dataset...")
    r_del_ds = requests.delete(f"{BASE}/api/datasets/{dataset_id}")
    assert r_del_ds.status_code == 200, f"Dataset deletion failed: {r_del_ds.text}"
    print(f"Response: {r_del_ds.json()}")
    
    # Verify dataset is removed from DB registry
    r_ds_list_check = requests.get(f"{BASE}/api/datasets/")
    datasets_check = r_ds_list_check.json()
    assert not any(d["id"] == dataset_id for d in datasets_check), "Dataset still found in DB list!"
    print("Dataset DB record successfully deleted.")
    
    # Verify dataset CSV file is removed from disk
    assert not os.path.exists(expected_file_path), f"Dataset CSV file still remains on disk: {expected_file_path}"
    print("Dataset CSV file successfully deleted from disk.")
    
    print("\n=== [SUCCESS] ALL DELETION AND CASCADE LIFECYCLE TESTS PASSED! ===")

if __name__ == "__main__":
    test_deletion_flow()
