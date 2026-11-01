"""Download the IBM Telco Customer Churn sample dataset (7,043 customers)."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    if not OUT.exists():
        OUT.write_bytes(urlopen(URL, timeout=60).read())
    print(f"{OUT} ({OUT.stat().st_size:,} bytes)")
