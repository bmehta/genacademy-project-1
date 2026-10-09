"""Loading, normalizing and filtering customer-feedback data."""
from __future__ import annotations

from datetime import date
from typing import IO, Iterable, Optional, Tuple, Union

import pandas as pd

REQUIRED_COLUMNS = ["date", "rating", "category", "review"]

# Common alternative header names, so a user's own CSV is likely to work.
ALIASES = {
    "date": ["date", "review_date", "reviewed_at", "created_at", "timestamp", "time"],
    "rating": ["rating", "stars", "score", "star_rating"],
    "category": ["category", "product", "product_category", "product_name", "department"],
    "review": ["review", "review_text", "text", "comment", "feedback", "body", "content"],
}


class DataError(ValueError):
    """Raised when a CSV cannot be turned into a usable feedback table."""


def sentiment_from_rating(rating: float) -> str:
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map columns to the canonical schema and add derived month/sentiment columns."""
    lowered = {c.strip().lower(): c for c in raw.columns}
    rename = {}
    missing = []
    for canonical, options in ALIASES.items():
        found = next((lowered[o] for o in options if o in lowered), None)
        if found is None:
            missing.append(canonical)
        else:
            rename[found] = canonical
    if missing:
        raise DataError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns like {', '.join(REQUIRED_COLUMNS)}; found {list(raw.columns)}."
        )

    df = raw.rename(columns=rename)[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["category"] = df["category"].astype("string").str.strip()
    df["review"] = df["review"].astype("string").str.strip()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise DataError("No valid rows after parsing dates and ratings.")

    df["category"] = df["category"].astype(str)
    df["review"] = df["review"].astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["sentiment"] = df["rating"].map(sentiment_from_rating)
    return df.sort_values("date").reset_index(drop=True)


def load_csv(source: Union[str, IO]) -> pd.DataFrame:
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # pandas raises many types for bad CSVs
        raise DataError(f"Could not read CSV: {exc}") from exc
    return normalize(raw)


def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[date, date]] = None,
    categories: Optional[Iterable[str]] = None,
    rating_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """Apply the sidebar filters. ``None`` means "don't filter on this"."""
    out = df
    if date_range is not None:
        start, end = date_range
        out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if categories is not None:
        out = out[out["category"].isin(list(categories))]
    if rating_range is not None:
        low, high = rating_range
        out = out[(out["rating"] >= low) & (out["rating"] <= high)]
    return out
