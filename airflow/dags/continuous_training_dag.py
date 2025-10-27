"""
Continuous Training DAG for automatic model retraining
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.bash import BashOperator

import sys
sys.path.append('/opt/airflow')

from loguru import logger


default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['your-email@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_new_data(**context):
    """Check if enough new data is available"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    query = """
        SELECT COUNT(*) 
        FROM plants_data 
        WHERE processed = TRUE 
        AND created_at > NOW() - INTERVAL '7 days'
    """
    
    result = pg_hook.get_first(query)
    new_data_count = result[0] if result else 0
    
    logger.info(f"New data count (last 7 days): {new_data_count}")
    
    # Threshold for retraining
    threshold = 100
    
    if new_data_count >= threshold:
        logger.info(f"✅ Sufficient new data ({new_data_count} >= {threshold}), triggering retraining")
        return 'retrain_model'
    else:
        logger.info(f"❌ Not enough new data ({new_data_count} < {threshold}), skipping retraining")
        return 'skip_retraining'


def check_model_performance(**context):
    """Check current model performance"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Get latest model metrics
    query = """
        SELECT accuracy, f1_score 
        FROM model_metrics 
        ORDER BY created_at DESC 
        LIMIT 1
    """
    
    result = pg_hook.get_first(query)
    
    if not result:
        logger.warning("No model metrics found, triggering retraining")
        return 'retrain_model'
    
    accuracy, f1_score = result
    
    logger.info(f"Current model - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
    
    # Thresholds
    min_accuracy = 0.90
    min_f1 = 0.85
    
    if accuracy < min_accuracy or f1_score < min_f1:
        logger.warning(f"⚠️ Model performance below threshold, triggering retraining")
        return 'retrain_model'
    else:
        logger.info(f"✅ Model performance is good")
        return 'check_new_data_task'


def save_model_metrics(**context):
    """Save model metrics to database after training"""
    # This would be called by the training script
    # Placeholder for now
    logger.info("Model metrics saved to database")
    return True


def notify_success(**context):
    """Notify team of successful retraining"""
    logger.success("🎉 Model retraining completed successfully!")
    # Add notification logic (Slack, email, etc.)
    return True


def skip_retraining():
    """Skip retraining"""
    logger.info("Skipping retraining - no trigger conditions met")
    return True


# Define DAG
with DAG(
    'continuous_training_pipeline',
    default_args=default_args,
    description='Continuous training pipeline with automatic triggers',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    start_date=datetime(2025, 10, 27),
    catchup=False,
    tags=['training', 'continuous', 'mlops'],
) as dag:
    
    # Check model performance first
    check_performance_task = BranchPythonOperator(
        task_id='check_model_performance',
        python_callable=check_model_performance,
        provide_context=True,
    )
    
    # Check for new data
    check_data_task = BranchPythonOperator(
        task_id='check_new_data_task',
        python_callable=check_new_data,
        provide_context=True,
    )
    
    # Retrain model
    retrain_task = BashOperator(
        task_id='retrain_model',
        bash_command='cd /opt/airflow && python src/training/train.py',
    )
    
    # Save metrics
    save_metrics_task = PythonOperator(
        task_id='save_model_metrics',
        python_callable=save_model_metrics,
        provide_context=True,
    )
    
    # Notify success
    notify_task = PythonOperator(
        task_id='notify_success',
        python_callable=notify_success,
        provide_context=True,
    )
    
    # Skip task
    skip_task = PythonOperator(
        task_id='skip_retraining',
        python_callable=skip_retraining,
    )
    
    # Define workflow
    check_performance_task >> [check_data_task, retrain_task]
    check_data_task >> [retrain_task, skip_task]
    retrain_task >> save_metrics_task >> notify_task
