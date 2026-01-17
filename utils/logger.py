import csv
import os
from datetime import datetime

class CsvLogger:
    def __init__(self, file_path='logs/access_log.csv'):
        self.file_path = file_path
        self.file_exists = os.path.isfile(self.file_path)
        if not self.file_exists:
            self.create_header()

    def create_header(self):
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Event', 'Message'])

    def log(self, event, message=''):
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, event, message])

# Example usage:
if __name__ == '__main__':
    # Initialize logger
    logger = CsvLogger()

    # Log events
    logger.log('Access Granted', 'User: admin')
    logger.log('Spoof Denied', 'Reason: Low liveness score')
