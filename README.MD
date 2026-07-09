# Fintel — Institutional Financial Intelligence Platform

A financial intelligence platform built to institutional standards —
designed for use inside large banking and asset management firms.
Covers NLP on SEC filings, knowledge graph construction, risk modelling,
portfolio analytics, and an LLM-powered analyst copilot.
---

## Quick start

```bash
cp .env.example .env                        # fill in your FRED API key
docker compose up -d postgres               # start database
docker compose run pipeline python run_phase1.py --step ingest --tickers AAPL JPM XOM
docker compose run pipeline python run_phase1.py --step clean  --tickers AAPL JPM XOM
docker compose run pipeline python run_phase1.py --step label  --tickers AAPL JPM XOM
docker compose run pipeline python run_phase1.py --step eda    --tickers AAPL JPM XOM
docker compose run pipeline python run_phase1.py --step check  # gate before Phase 2
```

---

## Project phases

| Phase | Status | What it builds |
|-------|--------|----------------|
| 1 | ✅ Active | Data foundation — ingestion, cleaning, labelling, EDA |
| 2 | 🔜 Next | Baseline + advanced models, metrics, evaluation |
| 3 | 🔜 Planned | Ensemble, error analysis, drift monitoring |
| 4 | 🔜 Planned | Analyst copilot (RAG + LLM interface) |

---

## Project structure

```
fintel/
├── Dockerfile                        multi-stage Python image
├── docker-compose.yml                single compose file for all phases
├── requirements.txt                  pinned dependencies
├── .env.example                      environment variable template
├── .gitignore                        excludes secrets, data, cache
├── config.py                         all constants — never hard-coded elsewhere
├── run_phase1.py                     master pipeline runner
│
├── src/
│   ├── ingestion/
│   │   ├── edgar_ingestion.py        SEC EDGAR filing pull
│   │   └── market_ingestion.py       yfinance prices + FRED macro
│   ├── cleaning/
│   │   └── cleaning.py               text cleaning + price gap handling
│   ├── labelling/
│   │   └── labelling.py              FinBERT labels + Cohen's κ + forward returns
│   ├── eda/
│   │   └── eda.py                    splits + look-ahead proof + 5 EDA experiments
│   └── db/
│       └── database.py               all Postgres read/write in one place
│
├── db/
│   ├── init/01_phase1_schema.sql     all Phase 1 tables, runs on first docker start
│   └── pgadmin_servers.json          optional pgAdmin config
│
├── data/                             gitignored — managed via Docker volumes
│   ├── raw/                          never modified after download
│   ├── processed/                    output of cleaning step
│   └── labels/                       FinBERT labels, annotation CSV, kappa result
│
├── outputs/                          EDA plots + JSON reports
└── logs/                             per-step log files + run manifests
```

---

## All commands

### Setup
```bash
# Copy and fill environment file
cp .env.example .env

# Get a free FRED API key at:
# https://fred.stlouisfed.org/docs/api/api_key.html
# Paste it into .env as FRED_API_KEY=your_key_here
```

### Docker
```bash
# Start database only
docker compose up -d postgres

# Start Jupyter notebook server (http://localhost:8888)
docker compose up -d notebook

# Run a specific pipeline step
docker compose run pipeline python run_phase1.py --step ingest
docker compose run pipeline python run_phase1.py --step clean
docker compose run pipeline python run_phase1.py --step label
docker compose run pipeline python run_phase1.py --step eda
docker compose run pipeline python run_phase1.py --step check

# Run on specific tickers (recommended for first test)
docker compose run pipeline python run_phase1.py --step all --tickers AAPL JPM XOM

# Run full 50-ticker universe
docker compose run pipeline python run_phase1.py --step all

# Tail logs live
docker compose logs -f pipeline

# Stop everything
docker compose down

# Full reset — wipes database volumes (clean slate)
docker compose down -v
```

### Manual annotation (Cohen's κ step)
```bash
# After running --step label, open this file and fill label_manual column:
# data/labels/annotation_sample.csv
# Values: positive / neutral / negative

# Then compute agreement:
docker compose run pipeline python -c \
  "from src.labelling.labelling import compute_label_agreement; compute_label_agreement()"
```

### Run without Docker (local dev)
```bash
pip install -r requirements.txt
python run_phase1.py --step all --tickers AAPL JPM XOM
```

---

## Phase 1 — Data Foundation

### What Phase 1 does and why it exists

Every modelling decision made in Phase 2 onwards depends on the quality of
data built here. If labels are unreliable, metrics are fictional. If train/test
splits leak future data, every accuracy number is inflated. Phase 1 exists to
build a foundation rigorous enough that every subsequent result is trustworthy.

This is the most skipped phase in student projects and the first thing a
senior engineer checks in a code review.

---

### Step 1 — Data ingestion

**What it does**

Pulls three data types in parallel:
- SEC EDGAR API → 10-K and 8-K filings for 50 S&P 500 companies
- yfinance → daily OHLCV price data, adjusted for splits and dividends
- FRED API → 7 macroeconomic series (CPI, GDP, VIX, yield curve, unemployment)

**Why these sources**

All three are free. SEC EDGAR is the official regulatory source — the same
data investment banks use. yfinance wraps Yahoo Finance and handles corporate
action adjustments automatically. FRED is the St. Louis Federal Reserve's
public data API, the standard source for academic and institutional macro data.

**Why adjusted prices specifically**

If you use unadjusted prices, a 2-for-1 stock split looks like a 50% price
drop. Your model learns a ghost signal that doesn't exist. Auto-adjusted prices
from yfinance account for splits and dividends so that all returns are
economically meaningful.

**Why not use a paid data vendor**

Bloomberg Terminal costs ~$25,000/year. Refinitiv is similar. For a Phase 1
build the free sources are sufficient. The architecture is vendor-agnostic —
swapping in a paid feed later means changing the ingestion module only.

**Why 50 tickers across 5 sectors**

Large enough to show cross-sector generalization. Small enough to inspect
manually, catch anomalies, and finish ingestion in a reasonable time.
Going straight to 500 tickers in Phase 1 is a common mistake — you spend
weeks debugging data issues you never noticed because you never looked.

**Why 3 years of history (2021–2024)**

Covers a full market cycle: post-COVID recovery, 2022 rate hike regime,
2023 recovery. Any shorter and you risk training only on one market regime.
Any longer and the oldest filings reflect a structurally different business
environment.

**Why raw files are never modified**

Raw data is saved to `data/raw/` and never touched again. Cleaned data goes
to `data/processed/`. This separation means you can re-run cleaning with
different parameters without re-downloading 50 companies worth of filings.
Reproducibility is not optional in an institutional setting.

**What you get out**

- `data/raw/filings/<TICKER>/<FORM>_<DATE>.json` — one file per filing
- `data/raw/prices/<TICKER>.csv` — full price history
- `data/raw/macro/<SERIES>.csv` — one file per macro series
- `logs/edgar_manifest_<timestamp>.json` — records exactly what was pulled, when

---

### Step 2 — Data cleaning

**What it does**

Two cleaning pipelines run in parallel:

Filing text cleaning: strips HTML tags, removes SEC boilerplate (table of
contents, "incorporated by reference", page numbers, item headers), tokenises
into sentences, filters sentences shorter than 8 words.

Price cleaning: detects missing trading days, forward-fills gaps up to 3
consecutive days, flags price moves greater than 5 standard deviations as
potential outliers.

**Why clean filings at all**

Raw SEC filings are HTML documents. Without cleaning, your NLP model sees
"&nbsp;&nbsp;ITEM 1A. RISK FACTORS&nbsp;" and "See Note 14 to the consolidated
financial statements" — legal boilerplate that appears identically across
every company and every year. It adds noise without signal. Removing it
reduces average token count from ~85k to ~40k per filing without losing
any analytically meaningful content.

**Why forward-fill price gaps instead of dropping them**

Dropping rows breaks the time series continuity that models depend on.
Forward-filling with a limit of 3 days handles public holidays and occasional
data feed outages — the standard approach in quantitative finance.

**Why flag outliers but not drop them**

A 5-sigma price move might be a data error or a real event (a takeover bid,
an earnings surprise, a regulatory announcement). Dropping it silently could
remove the most informative observations in the dataset. We flag and let the
modeller decide.

**What you get out**

- `data/processed/filings/<TICKER>/` — cleaned sentence lists per filing
- `data/processed/prices/<TICKER>.csv` — cleaned price series with return_1d
- `data/processed/filing_cleaning_stats.csv` — Experiment A data quality table
- `data/processed/price_cleaning_stats.csv` — Experiment B data quality table

**The data quality report**

These two CSVs become your "data quality" section in an interview.
You can show: average boilerplate removed (typically ~50%), average missing
price days per ticker (<1% is acceptable), number of outliers flagged.
Most candidates have no answer when asked "how did you ensure data quality."
This gives you a precise, quantified answer.

---

### Step 3 — Labelling

**What it does**

Three things: auto-labels every sentence using FinBERT, creates a random
stratified sample for manual annotation, and builds forward return labels
aligned to filing dates.

**Why FinBERT and not GPT-4**

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned specifically on
financial news and analyst reports. It achieves ~87% accuracy on the
Financial PhraseBank benchmark vs ~72% for generic BERT and ~65% for VADER.
GPT-4 would score similarly or higher but costs money per inference, runs
slowly at scale, and introduces non-determinism that makes results harder
to reproduce. For labelling 100,000 sentences, FinBERT running locally
on CPU is the right tool.

**Why not VADER**

VADER is a rule-based lexicon built for social media sentiment. It scores
"The company faces significant headwinds" as slightly positive because
"significant" appears in positive contexts in its training data. FinBERT
scores it correctly as negative because it was trained on financial text
where "headwinds" is a consistently negative signal.

**Why Cohen's κ and not raw agreement percentage**

Raw agreement can be high even when annotators are just guessing the majority
class. If 70% of sentences are neutral, two annotators randomly labelling
everything neutral agree 70% of the time — which tells you nothing about
label quality. Cohen's κ corrects for this chance agreement.

κ > 0.70 is the threshold from Landis and Koch (1977), still the standard
in NLP annotation literature. Below this, labels are too unreliable to
train a model on without introducing significant noise.

**Why forward returns as the prediction target**

We predict how the market reacts to a filing, not what the filing says.
The sentiment label tells us tone. The forward return label tells us
whether the market agreed. The gap between the two is where alpha lives.
A filing with negative tone but positive forward return is a potential
contrarian signal.

**Why a 5-day forward return window**

One day is too noisy — market reaction to a complex 10-K filing takes time
to be fully absorbed. One month is too slow — other events contaminate the
signal. Five trading days (one week) is the standard window in academic
event study literature for earnings and filing reactions.

**What you get out**

- `data/labels/auto_labels.csv` — per-sentence FinBERT labels + confidence scores
- `data/labels/annotation_sample.csv` — fill this manually for κ computation
- `data/labels/kappa_result.json` — κ score, gate pass/fail
- `data/labels/forward_returns.csv` — filing-level return labels

**Gate: κ must be ≥ 0.70 before proceeding to Phase 2**

---

### Step 4 — Train / val / test split design

**What it does**

Assigns every row a split label (train / val / test / embargo) based strictly
on chronological order. Runs the look-ahead bias proof experiment.

**Why chronological splits and not random**

Financial data is a time series. If you randomly shuffle and split, your
training set contains data from the future relative to your test set. The
model implicitly learns patterns that would not have been available at
prediction time. This is look-ahead bias — the most common mistake in
financial ML projects, and the first thing a quant interviewer will check.

**What the embargo period is**

A 20-trading-day gap between train end and val start. Returns autocorrelate
over short windows — today's news correlates with yesterday's price. Without
the embargo, information from the training period bleeds into the test period
through this autocorrelation, inflating metrics. 20 trading days (one calendar
month) is the standard in academic financial ML literature.

**The look-ahead bias proof experiment**

We train the same trivial model twice: once with a random split, once with
the time-ordered split. The random split gives inflated accuracy. The gap
between the two numbers is the magnitude of the bias. This experiment is
documented and saved to `outputs/look_ahead_proof.json`. In an interview
you can say: "We quantified the look-ahead bias at X accuracy points and
confirmed our pipeline eliminates it."

**Split dates used**

- Train: 2021-01-01 → 2022-12-31
- Embargo: Jan 2023 (~20 trading days)
- Val: 2023-02-01 → 2023-06-30
- Embargo: Jul 2023
- Test: 2023-08-01 → 2024-01-01

The test set is never touched until final model evaluation in Phase 2.

---

### Step 5 — Exploratory data analysis

**What it does**

Five experiments that describe your data before any model is trained.

**EDA 1 — Sentiment vs. forward return (Spearman ρ)**

Plots mean FinBERT sentiment score per filing against 5-day forward return.
Computes Spearman rank correlation.

Why this matters: even ρ = 0.05 is a meaningful signal in finance when
applied across a diversified portfolio. This single number justifies building
the entire NLP pipeline. If ρ is near zero, your labels and returns may not
be aligned correctly — catch this before training any model.

Why Spearman and not Pearson: financial returns have fat tails and outliers.
Spearman is rank-based and robust to these. Pearson assumes normality.

**EDA 2 — Filing length distribution**

Histogram of token counts across all 10-K and 8-K filings.

Why this matters: directly sets the chunk size for RAG in Phase 4. If median
10-K length is 60,000 tokens and you chunk at 512 tokens with 128-token stride,
you know you'll get roughly 450 chunks per filing. That determines your vector
database size, retrieval latency, and context window budget.

**EDA 3 — Sector sentiment heatmap**

Mean sentiment per sector per quarter, shown as a colour heatmap.

Why this matters: reveals whether your model is learning company-specific
signal or just sector-wide trends. If Energy sentiment tracks oil prices
and Technology sentiment tracks the NASDAQ, your model may be learning
macro rather than fundamental signal. This is a known confound you need
to control for in Phase 2.

**EDA 4 — Return autocorrelation (ACF plot)**

Autocorrelation function of daily returns up to lag 20.

Why this matters: confirms the efficient market hypothesis baseline.
Near-zero autocorrelation means price history alone is a weak predictor —
which justifies using NLP signal as the edge. If significant autocorrelation
exists at short lags, momentum is a confound that needs to be controlled.

**EDA 5 — Feature correlation matrix**

Pairwise correlations between all input features. Flags pairs above 0.85.

Why this matters: correlated features inflate apparent model performance
and cause coefficient instability. If FinBERT positive score and negative
score are perfectly anti-correlated (which they should be), you only need
one. If sentiment and price momentum are correlated, your model may attribute
explanatory power to the wrong feature.

---

## Key design principles

**Every raw file is immutable**
Raw downloaded files are never modified. Cleaning creates new files in
`data/processed/`. Re-running cleaning with new parameters never risks
corrupting source data.

**Single source of truth for configuration**
All constants live in `config.py`. No other file hard-codes a date, threshold,
ticker, or path. Changing the train/test split means changing two lines in
`config.py`, not hunting through six files.

**Postgres + CSV dual write**
Every pipeline result writes to both a CSV file and the Postgres database.
This means the pipeline works whether or not the database is running —
results are never lost. The database enables concurrent access from multiple
services in later phases.

**One docker-compose.yml for the whole project**
As phases are added, new services are appended to the same file. Commented
blocks for Phase 2+ services (Redis, MLflow, API, frontend) are already
present — uncomment them when you reach that phase. There will never be a
`docker-compose.phase2.yml`.

**Gate checks between phases**
Each phase ends with a checklist of required outputs. You do not proceed
to the next phase until all gates pass. This prevents building a Phase 2
model on Phase 1 data that was never properly validated.

---

## Interview talking points

**"How did you ensure your data was high quality?"**

"We ran three data quality experiments before training any model.
For filings, we measured boilerplate removal — typically ~50% of raw tokens
were noise. For prices, we checked missing trading days and outlier returns.
We saved a data quality report table that documents these numbers per ticker.
We also computed Cohen's κ between our auto-labels and manual annotations,
with a hard gate of κ ≥ 0.70 before proceeding."

**"How did you prevent look-ahead bias?"**

"We used strict chronological splits with a 20-trading-day embargo period
between train and validation sets. We also ran a proof experiment: training
the same model with a random split showed X% accuracy vs Y% with the time
split — a Z-point gap that vanishes with correct splitting. This proves
our pipeline is leak-free."

**"Why FinBERT over other sentiment models?"**

"FinBERT was fine-tuned specifically on financial text and achieves ~87%
accuracy on Financial PhraseBank vs ~65% for VADER and ~72% for generic
BERT. It correctly handles domain-specific language like 'headwinds' and
'guidance revision' that generic models misclassify. We also validated it
against manual annotations using Cohen's κ before accepting its labels."

**"What was your train/val/test split strategy?"**

"Chronological split with embargo. Train on 2021–2022, 20-day embargo,
validate on early 2023, 20-day embargo, test on late 2023 into 2024.
The test set was never touched until final evaluation. We chose a 5-day
forward return window based on academic event study literature for filing
reactions."

---

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FRED_API_KEY` | St. Louis Fed API key (free) | required |
| `POSTGRES_USER` | DB username | `fintel` |
| `POSTGRES_PASSWORD` | DB password | `fintel_dev_password` |
| `POSTGRES_DB` | Database name | `fintel` |
| `POSTGRES_HOST` | DB host (`postgres` in Docker) | `localhost` |
| `TEST_TICKERS` | Comma-separated tickers for quick test runs | `AAPL,JPM,XOM` |
| `JUPYTER_TOKEN` | Notebook login token | `fintel_notebook` |

---

## Dependencies and why each one

| Package | Version | Why |
|---------|---------|-----|
| `yfinance` | 0.2.40 | Free OHLCV data with auto-adjustment for corporate actions |
| `fredapi` | 0.5.1 | Official FRED API client for macro data |
| `transformers` | 4.40.1 | HuggingFace library — loads FinBERT |
| `torch` | 2.3.0 | FinBERT inference backend |
| `pandas` | 2.2.1 | Time series and tabular data handling |
| `scipy` | 1.13.0 | Spearman correlation, ACF computation |
| `scikit-learn` | 1.4.2 | Train/test split utilities, DummyClassifier for baseline |
| `sqlalchemy` | 2.0.29 | ORM — `DataFrame.to_sql()` writes to Postgres |
| `psycopg2-binary` | 2.9.9 | Postgres driver required by SQLAlchemy |
| `python-dotenv` | 1.0.1 | Loads `.env` into `os.environ` for local dev |
| `matplotlib` | 3.8.4 | EDA plots |
| `seaborn` | 0.13.2 | Heatmaps and styled plots |

---

## What comes next — Phase 2

Phase 2 adds to this repository:

- `src/models/` — baseline and advanced models per module
- `src/evaluation/` — metrics framework (F1, MCC, IC, Kupiec VaR backtest)
- `src/drift/` — PSI and KS drift detection
- Uncommenting `mlflow` and `redis` in `docker-compose.yml`
- `db/init/02_phase2_schema.sql` — model run and experiment tracking tables

Phase 2 gate requirement: baseline model trained and evaluated on the test
set with metrics documented before any complex model is attempted.