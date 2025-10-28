"""
Airflow DAG for data extraction and preprocessing
Downloads images from URLs and uploads to MinIO
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

import sys
sys.path.append('/opt/airflow')

from src.utils.helpers import download_image, validate_image, preprocess_image
from src.utils.s3_client import get_minio_client
from src.config import settings
from loguru import logger


default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def fetch_unprocessed_images(**context):
    """Fetch unprocessed images from database"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    query = """
        SELECT id, url_source, label 
        FROM plants_data 
        WHERE processed = FALSE 
        LIMIT 50
    """
    
    records = pg_hook.get_records(query)
    logger.info(f"Found {len(records)} unprocessed images")
    
    # Push to XCom
    context['ti'].xcom_push(key='unprocessed_images', value=records)
    return len(records)


def download_and_upload_images(**context):
    """Download images from URLs and upload to MinIO"""
    ti = context['ti']
    records = ti.xcom_pull(key='unprocessed_images', task_ids='fetch_unprocessed_images')
    
    if not records:
        logger.info("No images to process")
        return
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    minio = get_minio_client()  # Initialize MinIO client
    success_count = 0
    failed_count = 0
    
    for record_id, url_source, label in records:
        try:
            # Download image
            logger.info(f"Downloading image {record_id}: {url_source}")
            image = download_image(url_source)
            
            if image is None:
                logger.warning(f"Failed to download image {record_id}")
                failed_count += 1
                continue
            
            # Validate image
            if not validate_image(image):
                logger.warning(f"Image {record_id} failed validation")
                failed_count += 1
                continue
            
            # Preprocess
            image = preprocess_image(image, target_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            
            # Upload to MinIO
            object_name = f"{label}/{record_id:08d}.jpg"
            success = minio.upload_image(
                settings.S3_BUCKET_DATA,
                object_name,
                image
            )
            
            if success:
                # Update database
                url_s3 = f"s3://{settings.S3_BUCKET_DATA}/{object_name}"
                update_query = f"""
                    UPDATE plants_data 
                    SET processed = TRUE, 
                        url_s3 = '{url_s3}',
                        image_width = {image.size[0]},
                        image_height = {image.size[1]}
                    WHERE id = {record_id}
                """
                pg_hook.run(update_query)
                success_count += 1
                logger.success(f"✅ Processed image {record_id}")
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing image {record_id}: {e}")
            failed_count += 1
    
    logger.info(f"Processed: {success_count} success, {failed_count} failed")
    return success_count


def check_data_quality(**context):
    """Check data quality and balance"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    query = """
        SELECT label, COUNT(*) as count
        FROM plants_data
        WHERE processed = TRUE
        GROUP BY label
    """
    
    results = pg_hook.get_records(query)
    
    for label, count in results:
        logger.info(f"{label}: {count} images")
    
    # Check balance
    counts = [count for _, count in results]
    if len(counts) == 2:
        ratio = min(counts) / max(counts)
        logger.info(f"Class balance ratio: {ratio:.2f}")
        
        if ratio < 0.7:
            logger.warning("⚠️ Classes are imbalanced!")
    
    return True


# Define DAG
with DAG(
    'data_extraction_pipeline',
    default_args=default_args,
    description='Extract images from URLs and upload to MinIO',
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2025, 10, 27),
    catchup=False,
    tags=['data', 'extraction', 'mlops'],
) as dag:
    
    fetch_task = PythonOperator(
        task_id='fetch_unprocessed_images',
        python_callable=fetch_unprocessed_images,
        provide_context=True,
    )
    
    download_task = PythonOperator(
        task_id='download_and_upload_images',
        python_callable=download_and_upload_images,
        provide_context=True,
    )
    
    quality_task = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        provide_context=True,
    )
    
    # Define task dependencies
    fetch_task >> download_task >> quality_task
