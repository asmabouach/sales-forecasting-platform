from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta
import subprocess
import sys
import os

sys.path.insert(0, '/opt/airflow/src')  # Add src to path

def run_script(script_path):
    """Run a Python script and return output"""
    print(f"Running: {script_path}")
    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Output: {result.stdout[:500]}...")
        if result.stderr:
            print(f"Warnings: {result.stderr[:500]}...")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Script failed!")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise  # This will show the real error in Airflow UI

def write_update_timestamp():
    """Write last update timestamp to a file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    last_update_path = "/opt/airflow/sales_forecasting/output/last_update.txt"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(last_update_path), exist_ok=True)
    
    with open(last_update_path, "w", encoding="utf-8") as f:
        f.write(timestamp)
    
    print(f"✅ Last update timestamp written: {timestamp}")
    print(f"📁 File: {last_update_path}")
    return timestamp

def send_success_email():
    """Send success email"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    subject = f"✅ Pipeline Success - {timestamp}"
    
    html_content = f"""
    <h3>Sales Forecasting Pipeline Completed Successfully! 🎉</h3>
    
    <p><strong>Pipeline Details:</strong></p>
    <ul>
        <li><strong>DAG:</strong> sales_forecasting_pipeline</li>
        <li><strong>Status:</strong> ✅ SUCCESS</li>
        <li><strong>Completion Time:</strong> {timestamp}</li>
        <li><strong>Next Run:</strong> Scheduled for next Monday at 2 AM</li>
    </ul>
    
    <p><strong>Tasks Completed:</strong></p>
    <ol>
        <li>✅ Data Cleaning</li>
        <li>✅ Staging Population</li>
        <li>✅ Data Warehouse ETL</li>
        <li>✅ Weekly Forecasting</li>
        <li>✅ Monthly Forecasting</li>
        <li>✅ Timestamp Updated</li>
    </ol>
    
    <p>The dashboard has been updated with the latest forecasts.</p>
    
    <p style="color: #666; font-size: 0.9em; margin-top: 20px;">
        This is an automated notification from your Airflow pipeline.
    </p>
    """
    
    # Get recipient email from environment or use default
    to_email = os.getenv('MAILTRAP_TO', 'asma.bouach@gmail.com')
    
    print(f"📧 Sending success email to: {to_email}")
    send_email(to=to_email, subject=subject, html_content=html_content)
    
    return "Success email sent"

def send_failure_email(context):
    """Send failure email when pipeline fails"""
    task_instance = context['task_instance']
    failed_task = context['task'].task_id
    execution_date = context['execution_date']
    exception = context.get('exception', 'Unknown error')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    subject = f"❌ Pipeline Failed - {timestamp}"
    
    html_content = f"""
    <h3>Sales Forecasting Pipeline Failed! ⚠️</h3>
    
    <p><strong>Pipeline Details:</strong></p>
    <ul>
        <li><strong>DAG:</strong> sales_forecasting_pipeline</li>
        <li><strong>Status:</strong> ❌ FAILED</li>
        <li><strong>Failure Time:</strong> {timestamp}</li>
        <li><strong>Failed Task:</strong> {failed_task}</li>
        <li><strong>Execution Date:</strong> {execution_date}</li>
    </ul>
    
    <p><strong>Error Details:</strong></p>
    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
    {str(exception)}
    </pre>
    
    <p><strong>Tasks Completed Before Failure:</strong></p>
    <ol>
        <li>{"✅" if task_instance.previous_tasks_succeeded('cleaning') else "⏸️"} Data Cleaning</li>
        <li>{"✅" if task_instance.previous_tasks_succeeded('staging') else "⏸️"} Staging Population</li>
        <li>{"✅" if task_instance.previous_tasks_succeeded('dw') else "⏸️"} Data Warehouse ETL</li>
        <li>{"✅" if task_instance.previous_tasks_succeeded('weekly_forecast') else "⏸️"} Weekly Forecasting</li>
        <li>{"✅" if task_instance.previous_tasks_succeeded('monthly_forecast') else "⏸️"} Monthly Forecasting</li>
    </ol>
    
    <p style="color: #d9534f; font-weight: bold;">
        Action Required: Please check the Airflow logs for detailed error information.
    </p>
    """
    
    # Get recipient email from environment or use default
    to_email = os.getenv('MAILTRAP_TO', 'asma.bouach@gmail.com')
    
    print(f"📧 Sending failure email to: {to_email}")
    send_email(to=to_email, subject=subject, html_content=html_content)
    
    return "Failure email sent"

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email_on_failure': True,  # Airflow will send failure emails
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='sales_forecasting_pipeline',
    default_args=default_args,
    description='Complete sales forecasting pipeline',
    schedule_interval='0 2 * * 1',  # Monday 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sales', 'ml', 'forecasting'],
) as dag:

    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')
    

    PROJECT_ROOT = '/opt/airflow/sales_forecasting'

    cleaning = PythonOperator(
        task_id='cleaning',
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, 'src', 'cleaning', 'weekly_cleaning.py')],
    )
    
    staging = PythonOperator(
        task_id='staging',
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, 'src', 'staging', 'populate_staging.py')],
    )
    
    dw = PythonOperator(
        task_id='dw',
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, 'src', 'dw', 'populate_dw.py')],
    )
    
    weekly = PythonOperator(
        task_id='weekly_forecast',
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, 'src', 'ml', 'weekly_forecast.py')],
    )
    
    monthly = PythonOperator(
        task_id='monthly_forecast',
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, 'src', 'ml', 'monthly_forecast.py')],
    )

    write_timestamp = PythonOperator(
        task_id='write_timestamp',
        python_callable=write_update_timestamp,
    )
    
    success_email = PythonOperator(
        task_id='send_success_email',
        python_callable=send_success_email,
    )
    
    # Define pipeline flow
    start >> cleaning >> staging >> dw
    dw >> [weekly, monthly] >> write_timestamp >> success_email >> end