import logging
import os


def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'app.log'), level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')


def log_info(message):
    logging.info(message)


def log_error(message):
    logging.error(message)
