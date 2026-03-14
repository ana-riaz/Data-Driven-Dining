# Data Driven Dining


> **AI-Driven Marketing · Customer Profiling & Churn Prediction**
> End-to-end analytics pipeline that transforms raw restaurant POS data into actionable customer intelligence — with an AI-powered personalised email generator running fully locally via **Llama 3.2 (Ollama)**.

---

## Overview

This project analyses real restaurant transaction data across four phases to profile customers, identify behaviour patterns, predict churn risk, and generate targeted marketing emails — all from a single interactive Streamlit dashboard.

| Phase | What happens |
|-------|-------------|
| **I – Preprocessing** | Load & clean 12,545 transactions + 325 menu items, build composite join keys |
| **II – Pattern Analysis** | Classify spending (Premium / Standard / Economy), order timing (Morning / Mid-day / Evening) |
| **III – Customer Profiling** | One row per customer: lifetime spend, favourite picks, preferred timing, avg monthly orders |
| **IV – Churn Prediction** | RFM features (Recency, Frequency, Order Trend) → KMeans (k=4) → Regular / Occasional / New / Lost |

---
## Dashboard Snapshots

### Hero & KPI Metrics
![Hero and KPI metrics](assets/ss_dashboard.png)

---

### Spending Patterns & Order Timing
![Spending distribution and hourly order patterns](assets/ss_spending.png)

---

### Menu Intelligence — Top Items & Categories
![Best-selling items and category breakdown](assets/ss_menuPatterns.png)

---

### Churn Segmentation — RFM Clusters
![Customer segments, PCA cluster map and retention strategies](assets/ss_churnAnalytics.png)
![rfm](assets/ss_trends.png)
---

### AI Email Generator (Llama 3.2 via Ollama)
![Personalised AI-Powered Retention Email Generator](assets/ss_email.png)

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy |
| ML | scikit-learn (KMeans, PCA, StandardScaler) |
| Visualisation | Plotly Express, Plotly Graph Objects |
| Frontend | Streamlit |
| AI / LLM | Llama 3.2 via Ollama (OpenAI-compatible API) |

---

## Churn Segments & Retention Logic

| Segment | Criteria | Strategy |
|---------|----------|----------|
| **Regular** | Recency ≤ 20d & Frequency ≥ 7 | Loyalty rewards · birthday specials · early access |
| **Occasional** | Everything else | Personalised weekend deal · favourite picks highlight |
| **New** | Frequency ≤ 3 visits | Warm welcome email · 10% off next visit |
| **Lost** | Recency ≥ 80d & Frequency ≤ 6 | Win-back offer · 20% discount · time-limited |
