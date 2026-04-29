#!/usr/bin/env endometrial_env
"""
Preprocess the GEO matrices into model-ready tables.

This script does three things:
1) RNA-seq: gene_id x sample -> samples x genes
2) Microarray: probe_id x sample with Detection Pval columns -> samples x probes
3) Cross-platform alignment:
   - microarray probes -> gene symbols using GPL10558.annot.gz
   - RNA Ensembl IDs -> gene symbols using MyGeneInfo
   - intersect shared gene symbols and save matched matrices

Expected input layout:
project/
  data/
    geo/
      GSE234354/
        metadata_samples.csv
        supplementary/GSE234354_gene_count_matrix.txt.gz
      GSE234368/
        metadata_samples.csv
        supplementary/GSE234368_matrix_non-normalized.txt.gz
    GPL10558.annot.gz

Expected output layout:
project/
  data/
    processed/
      GSE234354/
      GSE234368/
      shared/
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("preprocess_geo")

RNA_ACCESSION = "GSE234354"
ARRAY_ACCESSION = "GSE234368"
RNA_MATRIX = "GSE234354_gene_count_matrix.txt.gz"
ARRAY_MATRIX = "GSE234368_matrix_non-normalized.txt.gz"
GPL10558_ANNOTATION = Path("data/GPL10558.annot.gz")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def extract_sample_name(title: str) -> str:
    """
    Extract the internal sample name from titles like:
      'endometrial biopsy, X210000'
      'endometrial biopsy, X210013_2'
    """
    if pd.isna(title):
        return ""
    text = str(title).strip().strip('"')
    return text.split(",")[-1].strip() if "," in text else text


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    required = {"gsm", "title"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns in {metadata_path}: {sorted(missing)}")

    meta = meta.copy()
    meta["gsm"] = meta["gsm"].astype(str).str.strip()
    meta["sample_name"] = meta["title"].apply(extract_sample_name).astype(str).str.strip()
    return meta


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def collapse_duplicate_columns(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate feature names by averaging across duplicates.
    """
    if expr.columns.duplicated().any():
        expr = expr.T.groupby(level=0).mean().T
    return expr


def log2_transform(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Log2 transform for RNA count-like data.
    """
    x = expr.apply(pd.to_numeric, errors="coerce")
    x = x.where(x >= 0, np.nan)
    return np.log2(x + 1.0)


def zscore_features(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each feature across samples.
    Input must be samples x features.
    """
    x = expr.apply(pd.to_numeric, errors="coerce")
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0).replace(0, np.nan)
    return ((x - mu) / sd).fillna(0.0)


def align_expression_and_metadata(expr: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only samples present in both expression and metadata, and order them consistently.
    Matching is done on sample_name extracted from title.
    """
    expr = expr.copy()
    meta = meta.copy()

    expr.index = expr.index.astype(str).str.strip()
    meta["sample_name"] = meta["sample_name"].astype(str).str.strip()

    common = [s for s in meta["sample_name"].tolist() if s in expr.index]
    if not common:
        raise ValueError("No overlap between metadata sample_name values and expression matrix rows.")

    expr = expr.loc[common]
    meta = meta[meta["sample_name"].isin(common)].copy()
    meta = meta.set_index("sample_name").loc[common].reset_index()
    return expr, meta


# -----------------------------------------------------------------------------
# RNA-seq
# -----------------------------------------------------------------------------

def load_rna_matrix(path: Path) -> pd.DataFrame:
    """
    RNA matrix format:
      gene_id | X210000 | X210013 | ...

    Returns samples x genes.
    """
    df = read_tsv(path)
    df = df.rename(columns={df.columns[0]: "gene_id"})
    df["gene_id"] = df["gene_id"].astype(str).str.strip()

    expr = df.set_index("gene_id").T
    expr.index = expr.index.astype(str).str.strip()
    expr.columns = [str(c).strip() for c in expr.columns]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = collapse_duplicate_columns(expr)
    expr = expr.dropna(axis=1, how="all")
    expr.index.name = "sample_name"
    return expr


# -----------------------------------------------------------------------------
# Microarray
# -----------------------------------------------------------------------------

def load_microarray_matrix(path: Path) -> pd.DataFrame:
    """
    Microarray matrix format:
      ID_REF | X210099 | Detection Pval | X210427 | Detection Pval.1 | ...

    We keep only the expression columns and drop detection p-value columns.
    Returns samples x probes.
    """
    df = read_tsv(path)
    df = df.rename(columns={df.columns[0]: "probe_id"})
    df["probe_id"] = df["probe_id"].astype(str).str.strip()

    expr_cols = [c for c in df.columns if c != "probe_id" and "Detection Pval" not in str(c)]
    if not expr_cols:
        raise ValueError(f"No expression columns found in {path}")

    expr = df[["probe_id"] + expr_cols].copy()
    expr = expr.set_index("probe_id").T
    expr.index = expr.index.astype(str).str.strip()
    expr.columns = [str(c).strip() for c in expr.columns]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = collapse_duplicate_columns(expr)
    expr = expr.dropna(axis=1, how="all")
    expr.index.name = "sample_name"
    return expr


# -----------------------------------------------------------------------------
# RNA mapping with MyGene
# -----------------------------------------------------------------------------

def _query_mygene(ids: Iterable[str], scopes, species: str = "human") -> pd.DataFrame:
    try:
        from mygene import MyGeneInfo
    except ImportError as e:
        raise SystemExit(
            "The 'mygene' package is required. Install it with: python -m pip install mygene"
        ) from e

    ids = [str(x).strip() for x in ids if str(x).strip()]
    if not ids:
        return pd.DataFrame(columns=["query", "symbol"])

    mg = MyGeneInfo()
    res = mg.querymany(
        ids,
        scopes=scopes,
        fields="symbol",
        species=species,
        verbose=False,
        as_dataframe=False,
    )
    df = pd.DataFrame(res)
    if df.empty or "symbol" not in df.columns or "query" not in df.columns:
        return pd.DataFrame(columns=["query", "symbol"])

    if "notfound" in df.columns:
        df = df[df["notfound"] != True].copy()

    df = df.dropna(subset=["symbol"]).copy()
    df["query"] = df["query"].astype(str).str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"] != ""].copy()
    df = df.drop_duplicates(subset=["query"], keep="first")
    return df[["query", "symbol"]]


def map_rna_ensembl_to_symbol(expr: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """
    Map Ensembl IDs to gene symbols using MyGeneInfo.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expr = expr.copy()

    cached = pd.DataFrame(columns=["query", "symbol"])
    if cache_path.exists():
        tmp = pd.read_csv(cache_path)
        if {"query", "symbol"}.issubset(tmp.columns):
            cached = tmp[["query", "symbol"]].copy()

    missing = [c for c in expr.columns if c not in set(cached["query"].tolist())]
    fresh = _query_mygene(missing, scopes="ensembl.gene") if missing else pd.DataFrame(columns=["query", "symbol"])

    mapping = pd.concat([cached, fresh], ignore_index=True).drop_duplicates(subset=["query"], keep="first")
    mapping.to_csv(cache_path, index=False)

    if mapping.empty:
        raise ValueError("No RNA Ensembl IDs could be mapped to gene symbols.")

    expr = expr.loc[:, ~expr.columns.duplicated()].copy()
    long = expr.T.reset_index().rename(columns={"index": "query"})
    merged = long.merge(mapping, on="query", how="inner")
    if merged.empty:
        raise ValueError("RNA mapping produced no overlapping features.")

    sample_cols = [c for c in merged.columns if c not in {"query", "symbol"}]
    merged[sample_cols] = merged[sample_cols].apply(pd.to_numeric, errors="coerce")
    out = merged.groupby("symbol", as_index=True)[sample_cols].mean().T
    out.index.name = "sample_name"
    out.columns = [str(c).strip() for c in out.columns]
    out = out.dropna(axis=1, how="all")
    return out


# -----------------------------------------------------------------------------
# Microarray annotation table (GPL10558)
# -----------------------------------------------------------------------------

def load_gpl10558_annotation(path: Path) -> pd.DataFrame:
    """
    Load the GPL10558 annotation table and return probe_id / gene_symbol.
    This version removes GEO comment lines before parsing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing GPL10558 annotation file: {path}")

    import gzip
    from io import StringIO

    opener = gzip.open if str(path).endswith(".gz") else open

    cleaned_lines = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            # Skip GEO metadata/comment lines.
            if line.startswith("!"):
                continue

            # Some headers may be comment-prefixed. If they contain tabs,
            # keep them after stripping the leading '#'.
            if line.startswith("#"):
                candidate = line[1:].lstrip()
                if "\t" in candidate:
                    cleaned_lines.append(candidate + "\n")
                continue

            # Keep only tabular lines.
            if "\t" in line:
                cleaned_lines.append(line + "\n")

    if not cleaned_lines:
        raise ValueError(f"No tabular annotation lines found in {path}")

    df = pd.read_csv(StringIO("".join(cleaned_lines)), sep="\t", dtype=str)
    df.columns = [str(c).strip().lstrip("#").strip() for c in df.columns]

    cols = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}

    probe_col = None
    for cand in ["id", "id_ref", "probe_id", "probeset_id", "probe id", "probe"]:
        if cand in lower:
            probe_col = lower[cand]
            break

    symbol_col = None
    for cand in ["gene symbol", "genesymbol", "symbol", "gene_symbol", "gene"]:
        if cand in lower:
            symbol_col = lower[cand]
            break

    if probe_col is None and len(cols) >= 1:
        probe_col = cols[0]
    if symbol_col is None:
        for c in cols:
            if re.search(r"gene\s*symbol|symbol", c, flags=re.I):
                symbol_col = c
                break

    if probe_col is None or symbol_col is None:
        raise ValueError(f"Could not identify probe and gene symbol columns in {path}. Columns: {cols[:20]}")

    mapping = df[[probe_col, symbol_col]].copy()
    mapping.columns = ["probe_id", "gene_symbol"]
    mapping["probe_id"] = mapping["probe_id"].astype(str).str.strip()
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.strip()

    mapping = mapping.replace({"gene_symbol": {"": np.nan, "nan": np.nan, "None": np.nan}})
    mapping = mapping.dropna(subset=["gene_symbol"])
    mapping = mapping[mapping["gene_symbol"] != ""].copy()
    mapping["gene_symbol"] = mapping["gene_symbol"].str.split(r"[;|,/ ]+").str[0].str.strip()
    mapping = mapping[mapping["gene_symbol"] != ""].copy()
    mapping = mapping.drop_duplicates(subset=["probe_id"], keep="first")
    return mapping


def map_microarray_probes_to_symbol(expr: pd.DataFrame, cache_path: Path, annotation_path: Path) -> pd.DataFrame:
    """
    Map microarray probe IDs to gene symbols using the GPL10558 annotation table.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expr = expr.copy()

    cached = pd.DataFrame(columns=["query", "symbol"])
    if cache_path.exists():
        tmp = pd.read_csv(cache_path)
        if {"query", "symbol"}.issubset(tmp.columns):
            cached = tmp[["query", "symbol"]].copy()

    missing = [c for c in expr.columns if c not in set(cached["query"].tolist())]

    fresh = pd.DataFrame(columns=["query", "symbol"])
    if missing:
        annot = load_gpl10558_annotation(annotation_path)
        fresh = annot.rename(columns={"probe_id": "query", "gene_symbol": "symbol"})[["query", "symbol"]]

    mapping = pd.concat([cached, fresh], ignore_index=True).drop_duplicates(subset=["query"], keep="first")
    mapping.to_csv(cache_path, index=False)

    if mapping.empty:
        raise ValueError("No microarray probes could be mapped to gene symbols.")

    expr = expr.loc[:, ~expr.columns.duplicated()].copy()
    long = expr.T.reset_index().rename(columns={"index": "query"})
    merged = long.merge(mapping, on="query", how="inner")
    if merged.empty:
        raise ValueError("Microarray mapping produced no overlapping features.")

    sample_cols = [c for c in merged.columns if c not in {"query", "symbol"}]
    merged[sample_cols] = merged[sample_cols].apply(pd.to_numeric, errors="coerce")
    out = merged.groupby("symbol", as_index=True)[sample_cols].mean().T
    out.index.name = "sample_name"
    out.columns = [str(c).strip() for c in out.columns]
    out = out.dropna(axis=1, how="all")
    return out


# -----------------------------------------------------------------------------
# Main processing steps
# -----------------------------------------------------------------------------

def process_rna(raw_dir: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_path = raw_dir / RNA_ACCESSION / "supplementary" / RNA_MATRIX
    metadata_path = raw_dir / RNA_ACCESSION / "metadata_samples.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing RNA matrix file: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing RNA metadata file: {metadata_path}")

    meta = load_metadata(metadata_path)
    expr_raw = load_rna_matrix(matrix_path)
    expr_raw, meta = align_expression_and_metadata(expr_raw, meta)

    out = out_dir / RNA_ACCESSION
    out.mkdir(parents=True, exist_ok=True)
    expr_raw.to_csv(out / "expression_raw_samples_by_features.csv")
    expr_log = log2_transform(expr_raw)
    expr_log.to_csv(out / "expression_log2_samples_by_features.csv")
    expr_z = zscore_features(expr_log)
    expr_z.to_csv(out / "expression_zscore_samples_by_features.csv")
    meta.to_csv(out / "metadata_aligned.csv", index=False)

    LOG.info("RNA raw matrix shape: %s", expr_raw.shape)
    return expr_raw, expr_log, expr_z


def process_microarray(raw_dir: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_path = raw_dir / ARRAY_ACCESSION / "supplementary" / ARRAY_MATRIX
    metadata_path = raw_dir / ARRAY_ACCESSION / "metadata_samples.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing microarray matrix file: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing microarray metadata file: {metadata_path}")

    meta = load_metadata(metadata_path)
    expr_raw = load_microarray_matrix(matrix_path)
    expr_raw, meta = align_expression_and_metadata(expr_raw, meta)

    out = out_dir / ARRAY_ACCESSION
    out.mkdir(parents=True, exist_ok=True)
    expr_raw.to_csv(out / "expression_raw_samples_by_features.csv")
    expr_z = zscore_features(expr_raw)
    expr_z.to_csv(out / "expression_zscore_samples_by_features.csv")
    meta.to_csv(out / "metadata_aligned.csv", index=False)

    LOG.info("Microarray raw matrix shape: %s", expr_raw.shape)
    return expr_raw, expr_z, meta


def build_shared_space(rna_log: pd.DataFrame, array_raw: pd.DataFrame, out_dir: Path, annotation_path: Path) -> None:
    shared_dir = out_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    rna_symbol = map_rna_ensembl_to_symbol(rna_log, shared_dir / "rna_mapping_cache.csv")
    array_symbol = map_microarray_probes_to_symbol(array_raw, shared_dir / "microarray_mapping_cache.csv", annotation_path)

    rna_symbol.to_csv(shared_dir / "rna_symbol_space.csv")
    array_symbol.to_csv(shared_dir / "microarray_symbol_space.csv")

    shared = sorted(set(rna_symbol.columns).intersection(array_symbol.columns))
    if not shared:
        raise ValueError("After mapping to gene symbols, no shared features were found.")

    (shared_dir / "shared_features.txt").write_text("\n".join(shared))

    rna_shared = rna_symbol[shared].copy()
    array_shared = array_symbol[shared].copy()
    rna_shared.to_csv(shared_dir / "rna_shared_log2.csv")
    array_shared.to_csv(shared_dir / "microarray_shared_log2.csv")

    rna_shared_z = zscore_features(rna_shared)
    array_shared_z = zscore_features(array_shared)
    rna_shared_z.to_csv(shared_dir / "rna_shared_zscore.csv")
    array_shared_z.to_csv(shared_dir / "microarray_shared_zscore.csv")

    LOG.info("Shared gene symbols: %d", len(shared))
    LOG.info("RNA shared matrix shape: %s", rna_shared_z.shape)
    LOG.info("Microarray shared matrix shape: %s", array_shared_z.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=project_root() / "data" / "geo")
    parser.add_argument("--out-dir", type=Path, default=project_root() / "data" / "processed")
    parser.add_argument("--skip-shared", action="store_true")
    parser.add_argument(
        "--gpl10558",
        type=Path,
        default=project_root() / GPL10558_ANNOTATION,
        help="Path to GPL10558.annot.gz",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Processing RNA-seq...")
    rna_raw, rna_log, rna_z = process_rna(args.raw_dir, args.out_dir)

    LOG.info("Processing microarray...")
    array_raw, array_z, array_meta = process_microarray(args.raw_dir, args.out_dir)

    if not args.skip_shared:
        LOG.info("Building shared gene-symbol space...")
        try:
            build_shared_space(rna_log, array_raw, args.out_dir, args.gpl10558)
        except Exception as e:
            LOG.warning("Shared space build skipped: %s", e)

    LOG.info("Done.")


if __name__ == "__main__":
    main()