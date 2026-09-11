"""
Central path configuration for the Lone Ranger pipeline.

All scripts import this so folder layout is defined ONCE. Scripts are run from
the project root (Lone Ranger/); this module locates the project root relative
to itself (it lives in src/), so the commands work regardless of the current
working directory.

Layout:
    Lone Ranger/
      data/raw/            raw NFL data (read-only): train/, supplementary_data.csv
      data/interim/        stepping-stone tables (rosters, role tables, layer counts)
      data/processed/      frozen + final analytical tables
      src/                 the pipeline (this file lives here)
      tests/               unit tests

Usage in a script:
    from paths import RAW, INTERIM, PROCESSED, TRAIN, SUPP, p
    frozen = pd.read_csv(PROCESSED / "opportunity_table_deep_v1.csv")
"""

from pathlib import Path

# src/ -> project root is one level up
ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

TRAIN = RAW / "train"                       # tracking files live here
SUPP = RAW / "supplementary_data.csv"       # supplementary table

# ensure output dirs exist (raw is never created here -- it must already hold data)
for d in (INTERIM, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)


def p(path_like):
    """Coerce a Path/str to str for libraries (e.g. glob) that want a string."""
    return str(path_like)
