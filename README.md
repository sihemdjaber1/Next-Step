# NextStep — Retail Location Intelligence

> **Open the right stores. Every time.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-nextstep-black?style=flat-square)](https://sihemdjaber1.github.io/Next-Step/)
![HTML](https://img.shields.io/badge/HTML-CSS-JS-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

---

## The Problem

Retail chains and franchise groups spend **€50,000+** and wait **6–12 weeks** for traditional location consultants — only to get a static report they can't reuse or test scenarios with.

## The Solution

NextStep is an AI-powered retail location optimizer that:
- Trains a demand model on your **historical client data**
- Simulates every possible client–store assignment
- Returns the **best-fit store locations** to maximize captured weekly revenue
- Delivers results in **under 5 minutes**

---

## How It Works

```
Upload 3 files → Map columns → Configure max stores → Get results
```

| Step | What happens |
|------|-------------|
| 1. Upload | Historical clients, potential clients, candidate sites (CSV or XLSX) |
| 2. Map | Tell the model which columns are demand, coordinates, features |
| 3. Run | k-NN demand model trains + greedy location optimizer runs |
| 4. Results | Interactive map, ranked site table, downloadable CSV |

---

## Features

- **Demand estimation** — k-NN model trained on your real historical data
- **Model validation** — Built-in backtest with MAE and R² score
- **Location optimizer** — Greedy algorithm selects best sites by captured demand
- **Interactive map** — Visual output with client and site markers
- **CSV export** — Download full results for further analysis
- **Demo mode** — Try with a built-in sample dataset (50 clients, 8 candidate sites)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript (vanilla) |
| ML | k-Nearest Neighbors (demand estimation) |
| Optimization | Greedy algorithm (location selection) |
| Mapping | Leaflet.js |
| Data parsing | SheetJS (XLSX), PapaParse (CSV) |
| Deployment | GitHub Pages |

---

## Pricing

| Plan | Price | Includes |
|------|-------|---------|
| Single Run | €490 | Full optimization + PDF report + CSV export |
| Growth Pack | €1,200 | 3 runs (save €270) + priority support |

---

## Roadmap

- [ ] Replace k-NN with regression models (linear, ridge) for more robust demand estimation
- [ ] Integrate Math Programming solver (PuLP / OR-Tools) for provably optimal solutions
- [ ] Multi-scenario comparison (run A vs run B)
- [ ] API access for enterprise clients
- [ ] Stripe payment integration

---

## Academic Context

Built as a Decision Analytics project at **ESSEC Business School** (M2, 2026), based on the OptiClean optimization case study.

> *"Your project looks very nice. I can see that you turned the OptiClean case into a nice app that could be useful in practice."*
> — Diego, Professor of Decision Analytics, ESSEC

---

## Author

**Sihem Djaber** — Founder, ESSEC Business School (Master in Management)  
[GitHub](https://github.com/sihemdjaber1) · [Live App](https://sihemdjaber1.github.io/Next-Step/)
