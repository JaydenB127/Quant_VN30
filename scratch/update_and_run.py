# -*- coding: utf-8 -*-
"""
Script to update experiment configuration and trigger a super fast 2-fold walk-forward validation run.
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
        # Get BTC_daily experiment
        res = await db.execute(select(Experiment).where(Experiment.id == uuid.UUID('f78f084c-2de9-483e-a101-ca118f795f01')))
        exp = res.scalar_one_or_none()
        if not exp:
            print("BTC_daily experiment not found!")
            return
            
        print(f"Updating config for experiment: {exp.name}")
        exp.config_json = {
            "skip_download": True,
            "fast": True,
            "quick": True,
            "initial_train_months": 72,
            "step_months": 24,
            "test_months": 24
        }
        await db.commit()
        print("Updated experiment config in DB successfully!")
        
        # Trigger run via API endpoint
        import requests
        print("\nTriggering run...")
        r = requests.post(f"http://127.0.0.1:8000/api/experiments/{exp.id}/run")
        if r.status_code != 200:
            print(f"Failed to trigger run: {r.status_code} - {r.text}")
            return
            
        run_info = r.json()
        run_id = uuid.UUID(run_info["run_id"])
        print(f"Triggered Run #{run_info['run_number']} | Run ID: {run_id}")
        
        # Poll status
        print("Polling run status...")
        for i in range(1, 31):
            await asyncio.sleep(2)
            async with db_manager.session_factory() as db2:
                res_run = await db2.execute(select(Run).where(Run.id == run_id))
                run = res_run.scalar_one_or_none()
                if not run:
                    print("Run not found!")
                    break
                print(f"[{i*2}s] Status: {run.status} | Error: {run.error_message}")
                if run.status in ("completed", "failed"):
                    # Check artifacts
                    res_art = await db2.execute(select(Artifact).where(Artifact.run_id == run_id))
                    arts = res_art.scalars().all()
                    print(f"\n=== Run finished with status: {run.status} ===")
                    print(f"Artifacts registered: {len(arts)}")
                    for art in sorted(arts, key=lambda a: a.name):
                        print(f"  - {art.name} ({art.artifact_type})")
                    break

if __name__ == "__main__":
    asyncio.run(main())
