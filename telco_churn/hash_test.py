from pathlib import Path
import hashlib

source = Path('/Users/marchelou/Documents/ML Ops/TD1/telco-churn/data/raw/telco.csv')  # Your input path.
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()