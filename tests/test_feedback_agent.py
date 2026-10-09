from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from feedback_agent.agent import ask, build_agent
from feedback_agent.data import DataError, apply_filters, load_csv, normalize
from feedback_agent.tools import (NO_DATA, aggregate, build_chart, describe_dataset, metrics_table,
                                  search_reviews)

SAMPLE = Path(__file__).parent.parent / "data" / "sample_reviews.csv"


@pytest.fixture(scope="module")
def df():
    return load_csv(str(SAMPLE))


class FakeToolModel(FakeMessagesListChatModel):
    """Scripted chat model that tolerates bind_tools so the real agent graph can run offline."""

    def bind_tools(self, tools, **kwargs):
        return self


# ------------------------------------------------------------------ data
def test_normalize_maps_aliases():
    raw = pd.DataFrame({"Stars": [5, 1], "Product": ["A", "B"], "Comment": ["good", "bad"],
                        "Review_Date": ["2026-01-02", "2026-02-03"]})
    out = normalize(raw)
    assert list(out["sentiment"]) == ["positive", "negative"]
    assert list(out["month"]) == ["2026-01", "2026-02"]


def test_normalize_reports_missing_columns():
    with pytest.raises(DataError, match="Missing required column"):
        normalize(pd.DataFrame({"rating": [1], "date": ["2026-01-01"]}))


def test_apply_filters_and_empty(df):
    only_watch = apply_filters(df, categories=["Smart Watch"])
    assert set(only_watch["category"]) == {"Smart Watch"}
    assert apply_filters(df, categories=[]).empty


# ------------------------------------------------------------------ tools
def test_describe_and_empty(df):
    assert "Reviews: 1800" in describe_dataset(df)
    assert describe_dataset(df.iloc[0:0]) == NO_DATA


def test_planted_story_smartwatch_battery_surge(df):
    watch = df[df["category"] == "Smart Watch"]
    before = watch[watch["date"] < "2026-06-01"]["rating"].mean()
    after = watch[watch["date"] >= "2026-06-01"]["rating"].mean()
    assert after < before - 0.5


def test_search_reviews_filters_and_caps(df):
    out = search_reviews(df, limit=100, keyword="battery", category="smart watch")
    assert "matching reviews" in out
    assert out.count("\n- ") <= 15
    assert search_reviews(df, keyword="zzzz-no-match") == NO_DATA


def test_metrics_table_values(df):
    t = metrics_table(df, "category")
    assert t["review_count"].sum() == len(df)
    assert t["pct_negative"].between(0, 100).all()
    split = metrics_table(df, "month", split_by="category")
    assert {"month", "category", "review_count", "avg_rating", "pct_negative"} <= set(split.columns)


def test_aggregate_and_chart(df):
    assert "pct_negative" in aggregate(df, "category")
    assert aggregate(df, "category", keyword="zzzz-no-match") == NO_DATA
    assert build_chart(df, "line", "month", "avg_rating", "category") is not None
    assert build_chart(df, "bar", "category", "review_count", keyword="zzzz-no-match") is None


# ------------------------------------------------------------------ agent wiring (offline)
def test_agent_runs_tools_and_returns_chart(df):
    llm = FakeToolModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "aggregate_metrics", "args": {"group_by": "category"}, "id": "c1"},
            {"name": "make_chart", "args": {"chart_type": "bar", "x": "category", "y": "avg_rating"}, "id": "c2"},
        ]),
        AIMessage(content="Smart Watch has the lowest rating."),
    ])
    result = ask(build_agent(lambda: df, llm=llm), "Which category is worst?")
    assert result.answer == "Smart Watch has the lowest rating."
    assert [s.name for s in result.steps] == ["aggregate_metrics", "make_chart"]
    assert "pct_negative" in result.steps[0].output
    assert len(result.charts) == 1 and result.charts[0].startswith("{")


def test_agent_tools_see_current_filters(df):
    llm = FakeToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "describe_dataset", "args": {}, "id": "c1"}]),
        AIMessage(content="done"),
    ])
    filtered = apply_filters(df, categories=["Speaker"])
    result = ask(build_agent(lambda: filtered, llm=llm), "what data?")
    assert "Speaker" in result.steps[0].output and "Smart Watch" not in result.steps[0].output
