#!/usr/bin/env python3
"""
MLflow Experiments Viewer - Display all experiments, runs, and metrics
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from mlflow.tracking import MlflowClient


def format_timestamp(timestamp_ms):
    """Convert timestamp in milliseconds to readable format"""
    if timestamp_ms:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    return 'N/A'


def list_experiments():
    """List all MLflow experiments"""
    
    print("="*80)
    print("🔬 MLFLOW EXPERIMENTS")
    print("="*80)
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"📊 MLflow URI: {mlflow_uri}\n")
    
    client = MlflowClient()
    
    try:
        experiments = client.search_experiments()
        
        if not experiments:
            print("ℹ️  No experiments found")
            return []
        
        exp_data = []
        for exp in experiments:
            exp_data.append([
                exp.experiment_id,
                exp.name,
                exp.lifecycle_stage,
                format_timestamp(exp.creation_time)
            ])
        
        print(tabulate(exp_data, 
                      headers=['Experiment ID', 'Name', 'Status', 'Created'],
                      tablefmt='grid'))
        
        return experiments
        
    except Exception as e:
        print(f"❌ Error listing experiments: {e}")
        return []


def list_runs_for_experiment(experiment_id, experiment_name):
    """List all runs for a specific experiment"""
    
    print(f"\n{'='*80}")
    print(f"🏃 RUNS FOR EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    client = MlflowClient()
    
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            print("ℹ️  No runs found for this experiment")
            return
        
        print(f"\n📊 Found {len(runs)} run(s)\n")
        
        for idx, run in enumerate(runs, 1):
            print(f"\n{'─'*80}")
            print(f"Run #{idx}: {run.info.run_name or run.info.run_id}")
            print(f"{'─'*80}")
            
            # Run info
            print(f"🔑 Run ID: {run.info.run_id}")
            print(f"📅 Started: {format_timestamp(run.info.start_time)}")
            print(f"⏱️  Ended: {format_timestamp(run.info.end_time)}")
            print(f"📊 Status: {run.info.status}")
            print(f"👤 User: {run.info.user_id}")
            
            # Parameters
            if run.data.params:
                print(f"\n📋 Parameters:")
                params_data = [[k, v] for k, v in sorted(run.data.params.items())]
                print(tabulate(params_data, headers=['Parameter', 'Value'], tablefmt='simple'))
            
            # Metrics
            if run.data.metrics:
                print(f"\n📈 Metrics:")
                metrics_data = [[k, f"{v:.6f}"] for k, v in sorted(run.data.metrics.items())]
                print(tabulate(metrics_data, headers=['Metric', 'Value'], tablefmt='simple'))
            
            # Tags
            if run.data.tags:
                print(f"\n🏷️  Tags:")
                tags_data = [[k, v] for k, v in sorted(run.data.tags.items()) 
                            if not k.startswith('mlflow.')]
                if tags_data:
                    print(tabulate(tags_data, headers=['Tag', 'Value'], tablefmt='simple'))
            
            # Artifacts
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                if artifacts:
                    print(f"\n📦 Artifacts:")
                    for artifact in artifacts:
                        print(f"   • {artifact.path} ({artifact.file_size} bytes)")
            except:
                pass
        
    except Exception as e:
        print(f"❌ Error listing runs: {e}")


def get_best_run(experiment_id, metric_name="val_accuracy"):
    """Get the best run based on a specific metric"""
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST RUN (by {metric_name})")
    print(f"{'='*80}")
    
    client = MlflowClient()
    
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        if not runs:
            print(f"ℹ️  No runs with metric '{metric_name}' found")
            return None
        
        best_run = runs[0]
        
        print(f"\n🎯 Best Run: {best_run.info.run_name or best_run.info.run_id}")
        print(f"🔑 Run ID: {best_run.info.run_id}")
        print(f"📊 {metric_name}: {best_run.data.metrics.get(metric_name, 'N/A')}")
        print(f"📅 Date: {format_timestamp(best_run.info.start_time)}")
        
        # Show all metrics for best run
        if best_run.data.metrics:
            print(f"\n📈 All Metrics:")
            metrics_data = [[k, f"{v:.6f}"] for k, v in sorted(best_run.data.metrics.items())]
            print(tabulate(metrics_data, headers=['Metric', 'Value'], tablefmt='grid'))
        
        return best_run
        
    except Exception as e:
        print(f"❌ Error finding best run: {e}")
        return None


def compare_runs(experiment_id):
    """Compare all runs in an experiment"""
    
    print(f"\n{'='*80}")
    print(f"📊 RUNS COMPARISON")
    print(f"{'='*80}")
    
    client = MlflowClient()
    
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            print("ℹ️  No runs to compare")
            return
        
        # Collect metrics for comparison
        metric_names = set()
        for run in runs:
            metric_names.update(run.data.metrics.keys())
        
        # Focus on key validation metrics
        key_metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_loss']
        available_metrics = [m for m in key_metrics if m in metric_names]
        
        if not available_metrics:
            print("ℹ️  No validation metrics to compare")
            return
        
        # Build comparison table
        comparison_data = []
        for run in runs:
            row = [
                run.info.run_name[:20] if run.info.run_name else run.info.run_id[:8],
                format_timestamp(run.info.start_time)
            ]
            for metric in available_metrics:
                value = run.data.metrics.get(metric, 0.0)
                row.append(f"{value:.4f}")
            comparison_data.append(row)
        
        headers = ['Run Name', 'Date'] + available_metrics
        print(f"\n{tabulate(comparison_data, headers=headers, tablefmt='grid')}")
        
    except Exception as e:
        print(f"❌ Error comparing runs: {e}")


def list_registered_models():
    """List all registered models"""
    
    print(f"\n{'='*80}")
    print(f"📦 REGISTERED MODELS")
    print(f"{'='*80}")
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    
    client = MlflowClient()
    
    try:
        registered_models = client.search_registered_models()
        
        if not registered_models:
            print("\nℹ️  No registered models found")
            print("💡 Tip: Run 'python3 scripts/mlflow_model_registry.py' to register your model")
            return
        
        for rm in registered_models:
            print(f"\n📦 Model: {rm.name}")
            print(f"   Description: {rm.description or 'N/A'}")
            print(f"   Created: {format_timestamp(rm.creation_timestamp)}")
            print(f"   Last Updated: {format_timestamp(rm.last_updated_timestamp)}")
            
            if rm.latest_versions:
                print(f"\n   Versions:")
                version_data = []
                for version in rm.latest_versions:
                    version_data.append([
                        version.version,
                        version.current_stage,
                        version.run_id[:8],
                        version.status
                    ])
                print("   " + tabulate(version_data, 
                                      headers=['Version', 'Stage', 'Run ID', 'Status'],
                                      tablefmt='simple').replace('\n', '\n   '))
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")


def main():
    """Main function"""
    
    print("\n" + "🔬"*40)
    print("MLflow Tracking - Complete View")
    print("🔬"*40 + "\n")
    
    # List all experiments
    experiments = list_experiments()
    
    if not experiments:
        print("\n💡 Run training script first: python3 scripts/train_advanced.py")
        return
    
    # Focus on dandelion-grass-classification experiment
    target_exp = None
    for exp in experiments:
        if exp.name == "dandelion-grass-classification":
            target_exp = exp
            break
    
    if target_exp:
        # List all runs
        list_runs_for_experiment(target_exp.experiment_id, target_exp.name)
        
        # Find best run
        get_best_run(target_exp.experiment_id, "val_accuracy")
        
        # Compare runs
        compare_runs(target_exp.experiment_id)
    
    # List registered models
    list_registered_models()
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Total Experiments: {len(experiments)}")
    print(f"🌐 MLflow UI: http://localhost:5001")
    print(f"💡 View detailed charts and compare runs in the UI")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
