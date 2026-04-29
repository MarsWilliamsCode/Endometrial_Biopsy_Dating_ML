#!/usr/bin/env endometrial_env
"""
Download and parse GEO series for the endometrial datasets.

Expected accessions:
- GSE234354: RNA-seq
- GSE234368: microarray

Outputs:
- data/geo/GSE234354/
- data/geo/GSE234368/

Each folder will contain:
- metadata_samples.csv
- supplementary files downloaded from GEO
- combined_expression_matrix.csv (if sample tables are available and combinable)
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

try:
    import GEOparse
except ImportError as e:
    raise SystemExit(
        "GEOparse is required. Install it with: python -m pip install GEOparse"
    ) from e


LOG = logging.getLogger("geo_download")


def safe_join(values) -> str:
    """Flatten GEO metadata lists into a readable string."""
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return "; ".join(str(v) for v in values if v is not None)
    return str(values)


def flatten_characteristics(metadata: Dict) -> Dict[str, str]:
    """
    Parse GEO 'characteristics_ch1' style fields into flatter columns.
    If a field appears multiple times, add numeric suffixes.
    """
    out: Dict[str, str] = {}
    for key, value in metadata.items():
        if not key.startswith("characteristics_ch1"):
            continue

        items = value if isinstance(value, (list, tuple)) else [value]
        for i, item in enumerate(items, start=1):
            text = str(item).strip()
            if ": " in text:
                k, v = text.split(": ", 1)
                k = k.strip().replace(" ", "_").replace("/", "_")
                col = k if k not in out else f"{k}_{i}"
                out[col] = v.strip()
            else:
                col = f"{key}_{i}"
                out[col] = text
    return out


def download_file(url: str, out_path: Path, timeout: int = 120) -> Path:
    """Download a file if it is not already present."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        LOG.info("Already exists: %s", out_path)
        return out_path

    # GEO often gives ftp:// links; requests cannot handle ftp directly.
    if url.startswith("ftp://"):
        url = "https://" + url[len("ftp://"):]

    LOG.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return out_path


def get_filename_from_url(url: str) -> str:
    """Extract a sensible filename from a GEO URL."""
    parsed = urlparse(url)
    name = Path(parsed.path).name

    # Some GEO links look like ...?file=XYZ.txt.gz
    if not name or name == "download":
        from urllib.parse import parse_qs

        qs = parse_qs(parsed.query)
        if "file" in qs and qs["file"]:
            name = qs["file"][0]

    return name or "downloaded_file"


def save_sample_metadata(gse, out_path: Path) -> pd.DataFrame:
    """
    Create a tidy metadata table for all GSM samples in a GSE object.
    """
    rows = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata if gsm.metadata is not None else {}
        flat = {
            "gsm": gsm_name,
            "title": safe_join(meta.get("title")),
            "source_name_ch1": safe_join(meta.get("source_name_ch1")),
            "organism_ch1": safe_join(meta.get("organism_ch1")),
            "platform_id": safe_join(meta.get("platform_id")),
            "series_id": safe_join(meta.get("series_id")),
            "type": safe_join(meta.get("type")),
            "relation": safe_join(meta.get("relation")),
            "supplementary_file": safe_join(meta.get("supplementary_file")),
        }

        flat.update(flatten_characteristics(meta))
        rows.append(flat)

    df = pd.DataFrame(rows)

    # Sort for readability
    sort_cols = [c for c in ["gsm", "platform_id", "title"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    df.to_csv(out_path, index=False)
    return df


def download_supplementary_files(gse, out_dir: Path) -> List[Path]:
    """
    Download supplementary files listed in the GSE metadata.
    """
    downloaded: List[Path] = []
    supp = gse.metadata.get("supplementary_file", []) if gse.metadata else []

    if not supp:
        LOG.info("No supplementary files listed for %s", gse.name)
        return downloaded

    supp_dir = out_dir / "supplementary"
    supp_dir.mkdir(parents=True, exist_ok=True)

    for url in supp:
        fname = get_filename_from_url(url)
        if not fname:
            continue
        dest = supp_dir / fname
        try:
            downloaded.append(download_file(url, dest))
        except Exception as e:
            LOG.warning("Failed to download %s: %s", url, e)

    return downloaded


def build_expression_matrix_from_gsm_tables(gse) -> Optional[pd.DataFrame]:
    """
    Try to combine per-sample GSM tables into one expression matrix.

    This works well when each GSM table has:
    - first column = feature/probe/gene ID
    - one value/intensity/count column

    If the series doesn't expose usable GSM tables, returns None.
    """
    frames = []

    for gsm_name, gsm in gse.gsms.items():
        tbl = getattr(gsm, "table", None)
        if tbl is None or tbl.empty:
            continue

        cols = list(tbl.columns)
        if len(cols) < 2:
            continue

        # Guess feature and value columns
        feature_col = cols[0]
        value_candidates = [
            c for c in cols
            if c.upper() in {
                "VALUE", "COUNT", "COUNTS", "INTENSITY", "EXPRESSION",
                "AVG_SIGNAL", "SIGNAL", "TARGET", "READS"
            }
        ]
        value_col = value_candidates[0] if value_candidates else cols[-1]

        s = tbl[[feature_col, value_col]].copy()
        s.columns = ["feature_id", gsm_name]
        s["feature_id"] = s["feature_id"].astype(str)
        s = s.dropna(subset=[gsm_name])

        # Collapse duplicates if any
        s = s.groupby("feature_id", as_index=True)[gsm_name].first().to_frame()
        frames.append(s)

    if not frames:
        return None

    expr = pd.concat(frames, axis=1, join="outer")
    expr.index.name = "feature_id"
    return expr


def download_and_parse(accession: str, base_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Download a GEO series, save metadata, supplementary files, and expression matrix if possible.
    """
    out_dir = base_dir / accession
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading GEO series %s", accession)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(out_dir), annotate_gpl=False)

    # Save GSE-level metadata snapshot
    gse_meta = pd.DataFrame(
        [{"field": k, "value": safe_join(v)} for k, v in (gse.metadata or {}).items()]
    )
    gse_meta.to_csv(out_dir / "series_metadata.csv", index=False)

    # Save sample-level metadata
    sample_meta = save_sample_metadata(gse, out_dir / "metadata_samples.csv")

    # Download supplementary files
    download_supplementary_files(gse, out_dir)

    # Build expression matrix from sample tables if possible
    expr = build_expression_matrix_from_gsm_tables(gse)
    if expr is not None:
        expr.to_csv(out_dir / "combined_expression_matrix.csv")

    return sample_meta, expr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/geo"),
        help="Output directory for downloaded GEO data",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    accessions = ["GSE234354", "GSE234368"]

    for acc in accessions:
        LOG.info("=" * 80)
        LOG.info("Processing %s", acc)
        sample_meta, expr = download_and_parse(acc, args.outdir)

        LOG.info("Saved metadata for %s samples: %d", acc, len(sample_meta))
        if expr is not None:
            LOG.info("Built expression matrix for %s with shape %s", acc, expr.shape)
        else:
            LOG.info("No combined expression matrix could be built from GSM tables for %s", acc)

    LOG.info("Done.")


if __name__ == "__main__":
    main()