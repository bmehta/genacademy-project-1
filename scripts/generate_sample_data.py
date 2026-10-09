"""Generate a SYNTHETIC product-review CSV (no real customers, no PII).

The data has a few planted stories so the agent has something to find:
  * Smart Watch battery complaints surge after a June 2026 "firmware update".
  * Headphones shipping delays spike in December 2025 (holiday rush).
  * Speaker ratings are steady and mostly positive.

Usage: python scripts/generate_sample_data.py
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
N_REVIEWS = 1800
START, END = date(2025, 10, 1), date(2026, 9, 30)
OUT = Path(__file__).resolve().parent.parent / "data" / "sample_reviews.csv"

CATEGORIES = {"Headphones": 0.30, "Smart Watch": 0.25, "Speaker": 0.25, "Charger": 0.20}

POSITIVE = [
    "Great sound quality and very comfortable.",
    "Exactly as described, works perfectly.",
    "Excellent value for the price, would buy again.",
    "Setup was easy and it feels well built.",
    "Love it, using it every day.",
    "Customer support answered fast and fixed my question.",
]
NEUTRAL = [
    "It's okay, does what it says but nothing special.",
    "Decent product, a few small annoyances.",
    "Average quality for the price.",
]
NEGATIVE = {
    "shipping": [
        "Arrived two weeks late and the box was damaged.",
        "Shipping took forever, tracking never updated.",
        "Package was delayed and the courier left it outside.",
    ],
    "battery": [
        "Battery barely lasts half a day since the update.",
        "Battery drains overnight even when I'm not using it.",
        "Used to last days, now I have to charge it constantly.",
    ],
    "quality": [
        "Stopped working after a month, feels cheap.",
        "The strap broke within weeks.",
        "Build quality is poor and the buttons stick.",
    ],
    "price": [
        "Way overpriced for what you get.",
        "Not worth the money, competitors are cheaper.",
    ],
    "support": [
        "Support never replied to my emails.",
        "Waited three weeks for a replacement that never came.",
    ],
}


def negative_theme_weights(category: str, d: date) -> dict[str, float]:
    w = {"shipping": 1.0, "battery": 0.6, "quality": 2.0, "price": 1.2, "support": 1.0}
    if category == "Smart Watch" and d >= date(2026, 6, 1):
        w["battery"] = 14.0
    if category == "Headphones" and date(2025, 12, 1) <= d <= date(2025, 12, 31):
        w["shipping"] = 12.0
    return w


def negative_probability(category: str, d: date) -> float:
    p = {"Headphones": 0.18, "Smart Watch": 0.22, "Speaker": 0.10, "Charger": 0.20}[category]
    if category == "Smart Watch" and d >= date(2026, 6, 1):
        p = 0.55
    if category == "Headphones" and date(2025, 12, 1) <= d <= date(2025, 12, 31):
        p = 0.45
    return p


def make_review(rng: random.Random, category: str, d: date) -> tuple[int, str]:
    r = rng.random()
    p_neg = negative_probability(category, d)
    if r < p_neg:
        weights = negative_theme_weights(category, d)
        theme = rng.choices(list(weights), weights=list(weights.values()))[0]
        text = rng.choice(NEGATIVE[theme])
        if rng.random() < 0.3:
            text += " " + rng.choice(NEUTRAL)
        return rng.choice([1, 1, 2, 2, 2]), text
    if r < p_neg + 0.12:
        return 3, rng.choice(NEUTRAL)
    text = rng.choice(POSITIVE)
    if rng.random() < 0.4:
        text += " " + rng.choice([t for t in POSITIVE if t != text])
    return rng.choice([4, 5, 5, 5]), text


def main() -> None:
    rng = random.Random(SEED)
    days = (END - START).days
    rows = []
    for _ in range(N_REVIEWS):
        category = rng.choices(list(CATEGORIES), weights=list(CATEGORIES.values()))[0]
        d = START + timedelta(days=rng.randint(0, days))
        rating, text = make_review(rng, category, d)
        rows.append({"date": d.isoformat(), "rating": rating, "category": category, "review": text})
    df = pd.DataFrame(rows).sort_values("date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} synthetic reviews to {OUT}")


if __name__ == "__main__":
    main()
