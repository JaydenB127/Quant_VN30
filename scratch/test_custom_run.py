# -*- coding: utf-8 -*-
"""
Test script to trigger and monitor an experiment run using a custom dataset.
"""
import asyncio
import sys
import os
import uuid

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

async def main():
    from db.session import DatabaseManager
    from sqlalchemy.future import select
    from db.models import Experiment, Dataset, Run, Artifact
    
    db_manager = DatabaseManager()
    
    async with db_manager.session_factory() as db:
        # Print all experiments and their datasets
        res = await db.execute(select(Experiment))
        experiments = res.scalars().all()
        print("=== EXPERIMENTS ===")
        for exp in experiments:
            stmt_ds = select(Dataset).where(Dataset.id == exp.dataset_id)
            res_ds = await db.execute(stmt_ds)
            ds = res_ds.scalar_one_or_none()
            ds_name = ds.name if ds else "None"
            print(f"ID: {exp.id} | Name: {exp.name} | Dataset: {ds_name} | Pipeline: {exp.pipeline_type}")
        
        # Find the experiment for BTC daily data if it exists, or create a new one
        btc_exp = None
        for exp in experiments:
            if "btc" in exp.name.lower():
                btc_exp = exp
                break
                
        if not btc_exp:
            print("\nCreating new experiment for BTC daily data...")
            # Find BTC dataset in DB
            stmt_ds = select(Dataset).where(Dataset.name.like("%BTC%"))
            res_ds = await db.execute(stmt_ds)
            btc_ds = res_ds.scalar_one_or_none()
            if not btc_ds:
                print("BTC dataset not found in DB!")
                return
            
            btc_exp = Experiment(
                name="BTC Forecasting Experiment",
                description="Forecast BTC returns using walk-forward",
                dataset_id=btc_ds.id,
                pipeline_type="finance_forecasting",
                config_json={"skip_download": True, "fast": True, "quick": True},
            )
            db.add(btc_exp)
            await db.commit()
            print(f"Created experiment {btc_exp.name} with ID {btc_exp.id}")
        
        # Trigger a new run via endpoint or directly
        print(f"\nTriggering run for experiment: {btc_exp.name} ({btc_exp.id})")
        import requests
        r = requests.post(f"http://127.0.0.1:8000/api/experiments/{btc_exp.id}/run")
        if r.status_code != 200:
            print(f"Failed to trigger run: {r.status_code} - {r.text}")
            return
        
        run_info = r.json()
        run_id = uuid.UUID(run_info["run_id"])
        print(f"Triggered Run #{run_info['run_number']} | Run ID: {run_id}")
        
        # Poll the run status
        print("Polling run status...")
        for _ in range(60):
            await asyncio.sleep(2)
            async with db_manager.session_factory() as db2:
                stmt_run = select(Run).where(Run.id == run_id)
                res_run = await db2.execute(stmt_run)
                run = res_run.scalar_one_or_none()
                if not run:
                    print("Run not found!")
                    break
                print(f"Status: {run.status} | Error: {run.error_message}")
                if run.status in ("completed", "failed"):
                    # Check artifacts
                    stmt_art = select(Artifact).where(Artifact.run_id == run_id)
                    res_art = await db2.execute(stmt_art)
                    arts = res_art.scalars().all()
                    print(f"\n=== Run finished with status: {run.status} ===")
                    print(f"Artifacts registered: {len(arts)}")
                    for art in arts:
                        print(f"  - {art.name} ({art.artifact_type}) | Size: {art.file_size_bytes} bytes | Key: {art.storage_key}")
                    break

if __name__ == "__main__":
    asyncio.run(main())
