"""Generate the claim package (email draft + PDF report) from the data folder.

    python -m scripts.make_claim_package                 # demo values, benchmark prices
    python -m scripts.make_claim_package --eeg 73.3 --ppa 70
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
ap.add_argument("--eeg", type=float, default=None, help="EEG reference value EUR/MWh (default: benchmark)")
ap.add_argument("--ppa", type=float, default=None, help="PPA price EUR/MWh (default: benchmark)")
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
    eeg_reference_value=args.eeg,
    ppa_price=args.ppa,
    price_basis=args.price_basis,
    waiting_period_h=args.waiting_hours,
)

result = run_assessment(args.data)
files = build_claim_package(result, info, args.out)
print("Subject:", files["email_subject"])
for k in ("email", "report"):
    print(f"{k:>7}: {files[k]}")
