import requests
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd

load_dotenv()
TOKEN = os.environ.get("SATNOGS_TOKEN")

HERE = Path(__file__).resolve().parent 

norad_id = "62391"
start_time = "2026-09-29T03:20:00Z"

print("Retrieving ")

r = requests.get(
    "https://network.satnogs.org/api/observations/",
    params={"status": "good", "norad_cat_id": norad_id, "start": start_time},
)

print("Requested URL:", r.url)
print("Request Status:", r.status_code)
df = pd.DataFrame(r.json())
print("Shape of Data:", df.shape)

df.to_csv(HERE / "lasarsat_observations.csv", index=False)