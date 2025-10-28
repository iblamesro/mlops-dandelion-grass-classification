"""
Script to check MLflow experiments and runs
"""

import mlflow
from mlflow.tracking import MlflowClient

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Create client
client = MlflowClient()

# List all experiments
print("=" * 80)
print("📊 MLFLOW EXPERIMENTS")
print("=" * 80)

experiments = client.search_experiments()
for exp in experiments:
    print(f"\n🔬 Experiment: {exp.name}")
    print(f"   ID: {exp.experiment_id}")
    print(f"   Lifecycle: {exp.lifecycle_stage}")
    print(f"   Artifact Location: {exp.artifact_location}")
    
    # Get runs for this experiment
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=10
    )
    
    print(f"\n   📈 Runs ({len(runs)}):")
    for run in runs:
        print(f"      - Run ID: {run.info.run_id}")
        print(f"        Name: {run.info.run_name}")
        print(f"        Status: {run.info.status}")
        print(f"        Start Time: {run.info.start_time}")
        
        # Print params
        if run.data.params:
            print(f"        Parameters:")
            for key, value in run.data.params.items():
                print(f"          • {key}: {value}")
        
        # Print metrics
        if run.data.metrics:
            print(f"        Metrics:")
            for key, value in run.data.metrics.items():
                print(f"          • {key}: {value:.4f}")
        
        print()

print("=" * 80)
print("✅ Check completed!")
print("=" * 80)
