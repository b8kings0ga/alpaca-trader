"""
Scheduling functionality for the Alpaca Trading Bot.
"""
import time
from datetime import datetime, timedelta
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from config import config
from src.logger import get_logger

logger = get_logger()

class TradingScheduler:
    """
    Class for scheduling the trading bot.
    """
    def __init__(self):
        """
        Initialize the scheduler.
        """
        self.scheduler = BackgroundScheduler()
        self.timezone = pytz.timezone(config.TIMEZONE)
        logger.info(f"Scheduler initialized with timezone: {config.TIMEZONE}")
        
    def start(self):
        """
        Start the scheduler.
        """
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")
        else:
            logger.warning("Scheduler is already running")
            
    def stop(self):
        """
        Stop the scheduler.
        """
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")
        else:
            logger.warning("Scheduler is not running")
            
    def add_job(self, job_func, job_id=None):
        """
        Add a job to the scheduler based on configuration.
        
        Args:
            job_func: Function to execute
            job_id (str): Job ID
            
        Returns:
            str: Job ID
        """
        if not job_id:
            job_id = f"trading_job_{int(time.time())}"
            
        # Remove existing job with the same ID if it exists
        self.remove_job(job_id)
        # Configure the job based on RUN_FREQUENCY
        if config.RUN_FREQUENCY == 'daily':
            # Parse the run time
            hour, minute = map(int, config.RUN_TIME.split(':'))
            
            # Add the job with a cron trigger
            self.scheduler.add_job(
                job_func,
                CronTrigger(
                    hour=hour,
                    minute=minute,
                    timezone=self.timezone
                ),
                id=job_id,
                replace_existing=True
            )
            
            logger.info(f"Added daily job at {config.RUN_TIME} {config.TIMEZONE}")
            
        elif config.RUN_FREQUENCY == 'hourly':
            # Add the job with an hourly trigger
            self.scheduler.add_job(
                job_func,
                CronTrigger(
                    minute=0,
                    timezone=self.timezone
                ),
                id=job_id,
                replace_existing=True
            )
            
            logger.info(f"Added hourly job at minute 0")
            
        elif config.RUN_FREQUENCY == 'minutes':
            # Get the interval in minutes
            interval = getattr(config, 'RUN_INTERVAL', 15)  # Default to 15 minutes if not specified
            
            # Add the job with a minutes interval trigger
            self.scheduler.add_job(
                job_func,
                'interval',
                minutes=interval,
                id=job_id,
                replace_existing=True
            )
            
            logger.info(f"Added job to run every {interval} minutes")
            logger.info(f"Added hourly job at minute 0")
            
        else:
            logger.error(f"Unknown run frequency: {config.RUN_FREQUENCY}")
            return None
            
        return job_id
        
    def remove_job(self, job_id):
        """
        Remove a job from the scheduler.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            bool: True if job was removed, False otherwise
        """
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception:
            # Job doesn't exist
            return False
            
    def get_next_run_time(self, job_id):
        """
        Get the next run time for a job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            datetime: Next run time
        """
        job = self.scheduler.get_job(job_id)
        if job:
            return job.next_run_time
        return None
        
    def add_market_hours_job(self, job_func, job_id=None):
        """
        Add a job that runs at market open.
        
        Args:
            job_func: Function to execute
            job_id (str): Job ID
            
        Returns:
            str: Job ID
        """
        if not job_id:
            job_id = f"market_open_job_{int(time.time())}"
            
        # Remove existing job with the same ID if it exists
        self.remove_job(job_id)
        
        # Parse the market open time
        hour, minute = map(int, config.MARKET_OPEN_TIME.split(':'))
        
        # Add the job with a cron trigger
        self.scheduler.add_job(
            job_func,
            CronTrigger(
                day_of_week='mon-fri',
                hour=hour,
                minute=minute,
                timezone=self.timezone
            ),
            id=job_id,
            replace_existing=True
        )
        
        logger.info(f"Added market hours job at {config.MARKET_OPEN_TIME} {config.TIMEZONE}")
        
        return job_id
        
    def run_once_now(self, job_func):
        """
        Run a job once immediately.
        
        Args:
            job_func: Function to execute
            
        Returns:
            str: Job ID
        """
        job_id = f"immediate_job_{int(time.time())}"
        
        self.scheduler.add_job(
            job_func,
            'date',
            run_date=datetime.now(self.timezone) + timedelta(seconds=1),
            id=job_id
        )
        
        logger.info(f"Added immediate job: {job_id}")
        
        return job_id