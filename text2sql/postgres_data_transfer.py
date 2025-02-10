import pandas as pd
from dotenv import load_dotenv
import os 

import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

# Database connection details
# DB_URI = os.getenv("DB_URI")

# PostgreSQL database connection parameters
db_params = {
    "dbname": "d8qdtku8976m7a",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": "5432"
}

# connection_kwargs = {
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD"),
#     "host": os.getenv("DB_HOST"),
#     "port": "5432",
#     "autocommit": True,
#     "prepare_threshold": 0,
# }

# read data 
df = pd.read_csv("data/processed/full_usage_data.csv")
df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)

# Define the CREATE TABLE query
create_table_query = """
CREATE TABLE IF NOT EXISTS smart_home_data (
    timestamp TIMESTAMP,
    use FLOAT,
    gen FLOAT,
    Dishwasher FLOAT,
    Furnace_1 FLOAT,
    Furnace_2 FLOAT,
    Home_office FLOAT,
    Fridge FLOAT,
    Wine_cellar FLOAT,
    Garage_door FLOAT,
    Kitchen_12 FLOAT,
    Kitchen_14 FLOAT,
    Kitchen_38 FLOAT,
    Barn FLOAT,
    Well FLOAT,
    Microwave FLOAT,
    Living_room FLOAT,
    temperature FLOAT,
    icon TEXT,
    humidity FLOAT,
    visibility FLOAT,
    summary TEXT,
    apparentTemperature FLOAT,
    pressure FLOAT,
    windSpeed FLOAT,
    cloudCover FLOAT,
    windBearing FLOAT,
    precipIntensity FLOAT,
    dewPoint FLOAT,
    precipProbability FLOAT,
    kitchen FLOAT,
    Furnace FLOAT
);
"""

# Define the INSERT query
insert_query = """
INSERT INTO smart_home_data (
    timestamp, use, gen, Dishwasher, Furnace_1, Furnace_2, Home_office, Fridge, Wine_cellar, Garage_door, 
    Kitchen_12, Kitchen_14, Kitchen_38, Barn, Well, Microwave, Living_room, temperature, icon, humidity, 
    visibility, summary, apparentTemperature, pressure, windSpeed, cloudCover, windBearing, precipIntensity, 
    dewPoint, precipProbability, kitchen, Furnace
) VALUES %s
"""


with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cur:
        # Create table if it does not exist
        cur.execute(create_table_query)
        conn.commit()

        # Convert DataFrame to list of tuples
        records = [tuple(x) for x in df.itertuples(index=False, name=None)]

        # Batch insert the data
        execute_values(cur, insert_query, records)
        conn.commit()
