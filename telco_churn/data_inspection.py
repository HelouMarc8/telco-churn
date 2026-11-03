import json
import sys
from pathlib import Path

import pandas as pd

from telco_churn.hash_test import file_sha256, source

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports" / "inspection_report.json"

df = pd.read_csv(source)
text_cols = df.select_dtypes(include=["object", "string"]).columns

report = {
    "source": {
        "path": str(source.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(source),
    },
    "environment": {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
    },
    "shape": {"rows": df.shape[0], "columns": df.shape[1]},
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "target_counts": {str(k): int(v) for k, v in df["Churn"].value_counts(dropna=False).items()},
    "target_proportions": {
        str(k): float(v) for k, v in df["Churn"].value_counts(normalize=True, dropna=False).items()
    },
    "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
    "missing_values": {c: int(n) for c, n in df.isna().sum().items()},
    "empty_or_whitespace_values": {c: int(df[c].str.strip().eq("").sum()) for c in text_cols},
}

REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2))
print(f"Report saved to {REPORT}")