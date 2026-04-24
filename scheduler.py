from apscheduler.schedulers.background import BackgroundScheduler
from app.ingest import run_ingestion

scheduler = BackgroundScheduler()

def start_scheduler():
    scheduler.add_job(run_ingestion, 'interval', hours=6)
    scheduler.start()
