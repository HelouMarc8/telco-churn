import hashlib
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    print(file_sha256(source))
