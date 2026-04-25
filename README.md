# Trading Systems Collection

## Dev/Creator = tubakhxn

---

## What is this?

This repository contains three core quantitative trading systems, each representing a fundamental market approach: trend following, momentum investing, and mean reversion. These systems are designed to be simple, visual, and practical, with clean matplotlib-based dashboards for analysis.

---

## 1. Trend Following System — MACD + EMA + ATR

### What is this project?

A trend-following strategy that captures sustained market moves using moving average crossovers and momentum confirmation, with ATR-based risk management.

**Features:**

* EMA crossover signals (short vs long trend)
* MACD confirmation for momentum strength
* ATR-based stop-loss for risk control
* Multi-panel dashboard (price, MACD, volume, equity curve, drawdown)

---

## 2. Multi-Asset Momentum Portfolio — Sector Rotation Engine

### What is this project?

A portfolio-level system that rotates capital into top-performing sectors using risk-adjusted momentum, rebalancing periodically to capture changing market leadership.

**Features:**

* Tracks multiple sector ETFs + gold
* Computes momentum adjusted by volatility
* Selects top-performing assets dynamically
* Portfolio allocation + equity curve visualization
* Correlation heatmap and return analysis

---

## 3. Mean Reversion System — RSI + Bollinger Bands

### What is this project?

A mean reversion strategy that trades price deviations from the mean, identifying overbought and oversold conditions using statistical indicators.

**Features:**

* RSI-based overbought/oversold detection
* Bollinger Bands for price deviation analysis
* Buy/sell signal generation
* Multi-panel dashboard (price, RSI, volume, performance)

---

## How to fork and run

1. **Fork this repository** using the GitHub interface.

2. **Clone your fork** to your local machine:

   ```sh
   git clone https://github.com/your-username/trading-systems-collection.git
   ```

3. **Navigate to the project folder**:

   ```sh
   cd trading-systems-collection
   ```

4. **Run any system**:

   ```sh
   python main.py
   ```

   Dependencies (pandas, numpy, matplotlib, yfinance) will auto-install if missing.

5. **View outputs** in the `outputs/` folder.

---

## Core Concepts Covered

* Trend Following
* Momentum Investing
* Mean Reversion
* Risk Management (ATR, drawdown)
* Portfolio Construction
* Quantitative Visualization

---

## Creator: tubakhxn

Built for practical quant learning — focusing on real systems, visual outputs, and fast implementation.
