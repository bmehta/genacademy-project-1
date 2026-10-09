# Customer Feedback Insights Agent

A Streamlit app with a LangChain/LangGraph agent that analyzes product-review CSVs. Explore the data on a
filterable dashboard, or ask questions in plain English and let the agent choose tools to answer them.

Built for The Gen Academy, *Mastering Agentic AI*, Week 1 (Path 3: build your own vibe-coded app).

## Features
- **Dashboard:** KPIs, rating distribution, ratings by category, monthly trends and a searchable review table,
  all driven by shared sidebar filters (date, category, rating). Empty filter results show a clear message.
- **Ask the Agent:** chat interface. Answers include charts and a "How I got this" panel showing every tool call.
- **Tools:** `describe_dataset`, `filter_reviews`, `aggregate_metrics`, `extract_themes` (LLM over a sample of
  reviews), `make_chart`. All tools respect the sidebar filters.
- **Bring your own CSV:** needs date, rating, category and review columns (common aliases such as `stars`,
  `product`, `comment` are mapped automatically).

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
streamlit run app.py
```
The dashboard works without an API key; the agent tab needs one.

## Sample data
`data/sample_reviews.csv` is **synthetic** (1,800 generated reviews, no real customers or PII), regenerated with
`python scripts/generate_sample_data.py`. It contains planted stories for the agent to find: Smart Watch battery
complaints surge from June 2026, and Headphones shipping complaints spike in December 2025.

Example questions: *"Which category has the lowest rating and why?"*, *"Show Smart Watch rating by month"*,
*"What are the top complaints in negative reviews?"*

## Project layout
```
app.py                       Streamlit UI (dashboard + chat)
feedback_agent/data.py       CSV loading, column mapping, filters
feedback_agent/tools.py      Pure analysis functions + LangChain tool wrappers
feedback_agent/agent.py      LangGraph ReAct agent, trace parsing (build_agent accepts a checkpointer for memory)
scripts/generate_sample_data.py
tests/                       pytest suite (agent graph tested offline with a scripted model)
```

## Tests
```bash
pytest -q
```

## Notes
- Use public or synthetic data only. Do not upload CUI or customer PII.
- Review text is treated as untrusted data (the prompt tells the model never to follow instructions inside reviews).
- AI output is a draft: verify important numbers against the dashboard before relying on them.
- Roadmap: conversation memory via a LangGraph checkpointer, then saved user preferences.
