"""Customer Feedback Insights Agent - Streamlit app.  Run with: streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from feedback_agent.agent import AgentResult, ask, build_agent
from feedback_agent.data import DataError, apply_filters, load_csv
from feedback_agent.tools import metrics_table

load_dotenv()
SAMPLE_PATH = Path(__file__).parent / "data" / "sample_reviews.csv"
EXAMPLE_QUESTIONS = [
    "Which category has the lowest average rating, and why?",
    "Did Smart Watch sentiment change over time? Show me a chart.",
    "What are the top complaints in negative reviews?",
    "How many reviews mention shipping, and when did they spike?",
]

st.set_page_config(page_title="Customer Feedback Insights Agent", page_icon="💬", layout="wide")


# --------------------------------------------------------------------------- data + filters
@st.cache_data(show_spinner=False)
def _load_sample() -> pd.DataFrame:
    return load_csv(str(SAMPLE_PATH))


def sidebar_data_and_filters():
    st.sidebar.header("Data")
    upload = st.sidebar.file_uploader("Upload a feedback CSV", type="csv",
                                      help="Needs date, rating, category and review columns.")
    st.sidebar.caption("Use public or synthetic data only - do not upload CUI or customer PII.")
    try:
        df = load_csv(upload) if upload else _load_sample()
    except DataError as exc:
        st.sidebar.error(str(exc))
        st.stop()
    st.sidebar.caption("Using your upload." if upload else "Using the bundled synthetic sample dataset.")

    st.sidebar.header("Filters")
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    picked = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    date_range = (picked[0], picked[1]) if len(picked) == 2 else (min_d, max_d)
    categories = st.sidebar.multiselect("Category", sorted(df["category"].unique()),
                                        default=sorted(df["category"].unique()))
    rating_range = st.sidebar.slider("Rating (stars)", 1, 5, (1, 5))
    return df, apply_filters(df, date_range, categories, rating_range)


# --------------------------------------------------------------------------- dashboard
def render_dashboard(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No reviews match the current filters. Try widening the date range, categories or ratings.")
        return
    neg = (df["sentiment"] == "negative").mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", f"{len(df):,}")
    c2.metric("Average rating", f"{df['rating'].mean():.2f} / 5 stars")
    c3.metric("Negative reviews", f"{neg:.1f}% (rated 1-2 stars)")

    left, right = st.columns(2)
    dist = df["rating"].astype(int).value_counts().sort_index().reset_index()
    dist.columns = ["rating", "reviews"]
    left.plotly_chart(px.bar(dist, x="rating", y="reviews", title="Rating distribution",
                             labels={"rating": "Stars", "reviews": "Reviews (count)"}), width="stretch")
    by_cat = metrics_table(df, "category")
    right.plotly_chart(px.bar(by_cat, x="category", y="avg_rating", title="Average rating by category",
                              labels={"avg_rating": "Average rating (1-5 stars)", "category": "Category"},
                              range_y=[1, 5]), width="stretch")

    monthly = metrics_table(df, "month", split_by="category")
    st.plotly_chart(px.line(monthly, x="month", y="avg_rating", color="category", markers=True,
                            title="Average rating by month",
                            labels={"avg_rating": "Average rating (1-5 stars)", "month": "Month"}),
                    width="stretch")
    st.plotly_chart(px.line(monthly, x="month", y="pct_negative", color="category", markers=True,
                            title="Negative reviews by month",
                            labels={"pct_negative": "Negative reviews (% of reviews)", "month": "Month"}),
                    width="stretch")

    st.subheader("Reviews")
    search = st.text_input("Search review text", placeholder="e.g. battery")
    view = df[df["review"].str.contains(search, case=False, regex=False)] if search else df
    st.caption(f"{len(view):,} reviews")
    st.dataframe(view[["date", "category", "rating", "review"]].sort_values("date", ascending=False),
                 width="stretch", hide_index=True)


# --------------------------------------------------------------------------- agent chat
def render_result(result: AgentResult) -> None:
    st.markdown(result.answer)
    for chart_json in result.charts:
        st.plotly_chart(pio.from_json(chart_json), width="stretch")
    if result.steps:
        with st.expander(f"How I got this ({len(result.steps)} tool call{'s' if len(result.steps) != 1 else ''})"):
            for i, step in enumerate(result.steps, 1):
                args = {k: v for k, v in step.args.items() if v is not None}
                st.markdown(f"**{i}. `{step.name}`** `{args}`")
                st.code(step.output[:1500], language="text")


def render_agent(df: pd.DataFrame) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.info("Set `ANTHROPIC_API_KEY` in a `.env` file (see `.env.example`) and restart to enable the agent.")
        return
    st.caption("The agent only sees the data left after your sidebar filters. AI answers are drafts - verify "
               "important numbers against the dashboard.")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    clicked = None
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, width="stretch"):
            clicked = q

    for entry in st.session_state.chat:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.markdown(entry["content"])
            else:
                render_result(entry["content"])

    question = st.chat_input("Ask about your customer feedback...") or clicked
    if not question:
        return
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                agent = build_agent(lambda: df)
                result = ask(agent, question)
            except Exception as exc:  # surface API/auth/network errors without crashing the app
                result = AgentResult(answer=f"Sorry, the agent hit an error: `{type(exc).__name__}: {exc}`")
        render_result(result)
    st.session_state.chat.append({"role": "assistant", "content": result})


def main() -> None:
    st.title("💬 Customer Feedback Insights Agent")
    st.caption("Explore review data on the dashboard, or ask the agent questions in plain English.")
    _, filtered = sidebar_data_and_filters()
    dash_tab, agent_tab = st.tabs(["Dashboard", "Ask the Agent"])
    with dash_tab:
        render_dashboard(filtered)
    with agent_tab:
        render_agent(filtered)


main()
