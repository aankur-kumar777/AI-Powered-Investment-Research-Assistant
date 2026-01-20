
An **industrial-grade AI investment research system** that transforms hours of manual analysis into seconds by combining:

* **LLM-powered document intelligence** (financial reports, filings, research PDFs)
* **Quantitative market modeling & backtesting**
* **Time-aware feature engineering**
* **Orchestrated pipelines with reproducibility and observability**

This project is designed to **stand up to real industry standards**: modular architecture, orchestration, experiment tracking, containerization, and scalable deployment.

---

## 🚀 What This Platform Does

### Core Capabilities

* 📄 Ingests unstructured financial documents (PDFs, text)
* 🧠 Extracts semantic signals using embeddings + LLMs
* 📊 Combines document-derived features with market data
* 🤖 Trains ML models to predict returns/direction
* 📈 Backtests strategies with realistic metrics
* 🔁 Runs reproducibly using Prefect orchestration
* 📦 Tracks experiments and artifacts via MLflow
* ⚙️ Deployable via Docker / Kubernetes

---

## 🧠 High-Level Architecture

```
ai-investment-research/
├── api/                # FastAPI application (ingest + query endpoints)
├── docs/               # Document loaders, chunkers, parsers
├── embeddings/         # Embedding models & vector store (Chroma)
├── llm/                # RAG pipelines & LLM utilities
├── pipelines/          # Quant + doc pipelines, Prefect flows
├── models/             # Trained ML models (joblib)
├── reports/            # CSV reports & backtest plots
├── config/             # Centralized YAML configuration
├── scripts/            # Local setup utilities
├── tests/              # Unit & integration tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔍 Key Design Principles

* **Separation of concerns**: ingestion, intelligence, modeling, orchestration
* **Reproducibility**: Prefect + MLflow
* **Scalability**: parallel multi-ticker execution
* **Extensibility**: swap vector DBs, LLMs, models
* **Production readiness**: CI, Docker, K8s manifests

---

## 🛠️ Tech Stack

### Core

* **Python 3.10**
* **FastAPI** – API layer
* **Prefect 2.x** – workflow orchestration
* **MLflow** – experiment tracking

### AI / ML

* **Sentence-Transformers** – embeddings
* **LangChain** – LLM orchestration
* **XGBoost** – predictive modeling
* **VectorBT** – backtesting engine
* **ChromaDB** – vector store

### Data & Infra

* **yfinance** – market data
* **PostgreSQL** – metadata / persistence
* **Redis** – caching / queues
* **Docker / Docker Compose**
* **Kubernetes (optional)**

---

## 📦 Installation (Local)

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ai-investment-research
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> System dependencies (Linux):

```bash
sudo apt-get install poppler-utils tesseract-ocr
```

---

## 🐳 Run with Docker (Recommended)

```bash
docker-compose up --build
```

Services started:

* FastAPI → `http://localhost:8000`
* PostgreSQL
* Redis
* Persistent Chroma vector store

Health check:

```bash
curl http://localhost:8000/health
```

---

## 📄 Document Ingestion

### Supported Inputs

* PDFs (earnings, filings, reports)
* Plain text files

### Flow

1. Upload document
2. Chunk + clean text
3. Generate embeddings
4. Store in ChromaDB
5. Ready for retrieval during modeling

---

## 🤖 Quant + Doc Pipeline

### What Happens

1. Download market data (yfinance)
2. Compute technical indicators
3. Retrieve relevant document embeddings
4. Generate document-based features (sentiment, relevance)
5. Train ML model (XGBoost)
6. Generate trading signals
7. Backtest strategy
8. Save artifacts

### Run Single Ticker

```bash
python -m pipelines.quant_run_with_docs --ticker AAPL --period 1y
```

Outputs:

```
reports/AAPL/
├── metrics.csv
├── cumulative_returns.png
└── predictions.csv
models/
└── aapl_xgb_with_docs.joblib
```

---

## 🔁 Orchestration with Prefect

### Single-Ticker Flow

```bash
python -m pipelines.prefect_flows --ticker AAPL
```

### Multi-Ticker (Parallel)

```bash
python -m pipelines.prefect_multi_ticker --tickers AAPL,MSFT,GOOG
```

* Each ticker runs independently
* Failures are isolated
* Can scale horizontally with Prefect agents

---

## 📊 Experiment Tracking (MLflow)

Start MLflow UI:

```bash
mlflow ui
```

View:

* Model metrics
* Backtest results
* CSV & PNG artifacts
* Model versions

---

## ☁️ Deployment (Kubernetes)

### Build Image

```bash
docker build -t ai-invest-research:latest .
```

### Apply Manifests

```bash
kubectl apply -f k8s/
```

Supports:

* Horizontal scaling
* Persistent vector DB
* Secure secrets via K8s secrets

---

## 🧪 Testing

```bash
pytest
```

CI runs automatically via GitHub Actions:

* Unit tests
* Lint checks

---

## 🔐 Configuration

Centralized in:

```
config/defaults.yaml
```

Override via environment variables:

* `OPENAI_API_KEY`
* `DATABASE_URL`
* `MLFLOW_TRACKING_URI`
* `CHROMA_PERSIST_DIR`

---

## 📈 Example Use Cases

* Equity research automation
* Earnings impact analysis
* News-driven signal generation
* Long/short factor modeling
* AI-assisted portfolio screening

---

## 🧩 Extending the Platform

Easy extensions:

* Replace Chroma → Qdrant / Pinecone
* Replace XGBoost → deep learning models
* Add real-time streaming ingestion
* Add model drift detection
* Add portfolio-level optimization

---

## ⚠️ Disclaimer

This project is for **research and educational purposes**.
It does **not constitute financial advice**.
Trading involves risk.

---

## 📌 Status

✅ Architecture complete
✅ Fully runnable pipelines
✅ Orchestration + tracking
✅ Production-ready scaffolding

---

## 📬 Next Planned Enhancements

* Monitoring (Prometheus + Grafana)
* LLM cost tracking
* Online inference API
* Portfolio-level execution engine

---

**Built for serious AI-first investment research.**
