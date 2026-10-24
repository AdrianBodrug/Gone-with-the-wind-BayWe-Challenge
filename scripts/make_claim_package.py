"""Generate the claim package (email draft, PDF report, CSV) from the data folder.

    python -m scripts.make_claim_package                 # demo values
    python -m scripts.make_claim_package --price 73.5 --waiting-hours 24
"""
import argparse
import warnings

from src.assessment import run_assessment
from src.report import build_claim_package
from src.schema import ClaimInfo

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="outputs")
ap.add_argument("--valuation", choices=["market", "eeg", "fixed"], default=None,
                help="market = live day-ahead prices; eeg/fixed need --price")
ap.add_argument("--price", type=float, default=None, help="EUR/MWh (EEG reference value or PPA price)")
ap.add_argument("--price-basis", default="")
ap.add_argument("--waiting-hours", type=float, default=None, help="policy time deductible in hours")
args = ap.parse_args()

# Demo values - clearly fictional. In the app these come from a form.
info = ClaimInfo(
    wind_farm_name="Wind Farm T (demo)",
    operator_name="Demo Operator GmbH",
    insurer_name="Demo Insurance AG",
    policy_number="DEMO-0000-00",
    contact_name="Asset Operations Team",
    contact_email="operations@example.com",
    turbine_model="4.5 MW class",
    valuation_mode=args.valuation,
    price_eur_per_mwh=args.price,
    price_basis=args.price_basis,
    waiting_period_h=args.waiting_hours,
)

result = run_assessment(args.data)
files = build_claim_package(result, info, args.out)
print("Subject:", files["email_subject"])
for k in ("email", "report", "data", "valuation", "zip"):
    if k in files:
        print(f"{k:>9}: {files[k]}")
