import os
from dotenv import load_dotenv

os.environ['RUN_TIMEZONE_CHECK'] = '0'


from ingest import ingest_data

load_dotenv()

if __name__ == "__main__":
    print("Initializing database...")
    ingest_data()
    print("Database initialized")