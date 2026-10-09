# 🤖 Advanced AI Forex Trading Bot

> **An advanced Python-based algorithmic Forex trading system that combines MetaTrader 5, Exness market data, machine learning models, technical analysis, ATR-based trade management, multi-symbol execution, threshold optimization, forward testing, risk management, and automated trade execution.**

---

## 📌 Overview

This repository contains the source code, trained machine learning models, market datasets, analysis tools, backtesting/forward-testing utilities, and live trading components of an **advanced automated Forex trading system** built with Python.

The system is designed to connect directly to **MetaTrader 5 (MT5)**, retrieve market data from an **Exness trading account**, process and engineer trading features, generate machine learning predictions, apply multiple entry and tradeability filters, calculate position sizes, and automatically execute and manage trades.

Rather than being a single Python script, this project is structured as a complete trading pipeline:

```text
Market Data
     │
     ▼
MetaTrader 5 / Exness
     │
     ▼
Historical & Live CSV Data
     │
     ▼
Feature Engineering
     │
     ▼
Machine Learning Models
     │
     ▼
Prediction
     │
     ▼
Threshold / Signal Validation
     │
     ▼
Entry Filtering
     │
     ├── Spread Filter
     ├── Correlation Check
     ├── Market Condition Checks
     ├── Trading Session Checks
     └── Other Trade Filters
     │
     ▼
Risk Management
     │
     ├── Lot Size Calculation
     ├── Stop Loss
     ├── Take Profit
     └── Position Management
     │
     ▼
MetaTrader 5
     │
     ▼
Automated Trade Execution
     │
     ▼
Trade Logging & Monitoring
```

The repository therefore covers both the **research/development side** and the **live execution side** of an automated Forex trading system.

---

# ⚠️ Important Disclaimer

This project is provided for **educational, research, software-development, and algorithmic-trading experimentation purposes**.

It is **not financial advice**.

Automated Forex trading involves substantial risk. Machine learning predictions do not guarantee profitable trades, and historical/backtest/forward-test performance does not guarantee future results.

Before connecting this software to a live account:

- Test the system extensively.
- Verify every trading parameter.
- Verify broker symbol specifications.
- Test the MT5 connection.
- Test order execution.
- Test stop-loss and take-profit behavior.
- Test position sizing.
- Test failure/reconnection scenarios.
- Test the system on a demo account.
- Never expose your broker credentials or API/login information publicly.
- Never assume that historical performance will continue in live markets.

Use live trading only when you fully understand the risks and the behavior of the software.

---

# 🚀 Main Features

## 🧠 Machine Learning-Based Trading

The system uses trained machine learning models to analyze market data and generate trading predictions.

The repository contains trained models for multiple Forex symbols, allowing the system to process different markets independently.

The architecture supports:

- Model training
- Model loading
- Model prediction
- Threshold optimization
- Historical analysis
- Forward testing
- Realistic forward testing
- Live prediction
- Symbol-specific models
- Model information inspection

---

# 📊 Multi-Symbol Trading

The trading engine is designed to run multiple Forex symbols simultaneously.

The primary symbol configuration includes:

```python
symbols = [
    "AUDCAD",
    "AUDUSD",
    "CADJPY",
    "EURJPY",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "NZDJPY",
    "NZDUSD",
    "USDCHF",
    "USDJPY"
]
```

For the Exness Cent environment, the system converts the symbols to the broker-specific `c` suffix:

```python
symbols = [symbol + 'c' for symbol in symbols]
```

Result:

```text
AUDCADc
AUDUSDc
CADJPYc
EURJPYc
EURUSDc
GBPJPYc
GBPUSDc
NZDJPYc
NZDUSDc
USDCHFc
USDJPYc
```

The repository's trained-model and dataset architecture is designed around multiple symbols, allowing the trading engine to operate on markets independently rather than relying on a single currency pair.

---

# 🏦 MetaTrader 5 + Exness

The live trading system uses **MetaTrader 5** as the trading interface.

Market data is obtained from an **Exness-connected MT5 environment**, while MT5 is responsible for communicating with the broker environment for market information and trade execution.

The architecture therefore separates the strategy logic from the broker interface.

```text
Python
   │
   ▼
MetaTrader5 Python API
   │
   ▼
MT5 Terminal
   │
   ▼
Exness
   │
   ├── Market Data
   ├── Account Information
   ├── Symbol Information
   └── Trade Execution
```

---

# 📁 Repository Structure

The project is organized into three major areas:

```text
ADVANCED-FOREX-TRADING-BOT/
│
├── ALL MODELS/
│   │
│   ├── Trained model files
│   ├── Symbol-specific models
│   └── Model artifacts
│
├── CSV FILES/
│   │
│   ├── AUDCAD...
│   ├── AUDUSD...
│   ├── CADJPY...
│   ├── EURJPY...
│   ├── EURUSD...
│   ├── GBPJPY...
│   ├── GBPUSD...
│   ├── NZDJPY...
│   ├── NZDUSD...
│   ├── USDCHF...
│   ├── USDJPY...
│   └── Other MT5 market datasets
│
├── PY FILES/
│   │
│   ├── ATR_ALL_FIND_THRESHOLD.py
│   ├── ATR_ANALYSIS_&_TEST.py
│   ├── ATR_FIND_THRESHOLD.py
│   ├── ATR_FORWARD_TEST_MODEL.py
│   ├── ATR_REALISTIC_FOWARDT.py
│   ├── ATR_TRADE_HIGH_NXT.py
│   ├── ATR_TRAIN_MODEL.py
│   ├── CONNECT_MT5.py
│   ├── FOREX_TRADING.py
│   ├── func.py
│   ├── GET_MT5_DATASET.py
│   ├── LOAD_MODEL_INFOS.py
│   ├── SYMBOL_DATA.py
│   └── SYMBOL_POINTS.py
│
└── README.md
```

---

# 🧠 1. ALL MODELS

The `ALL MODELS` directory contains the trained machine learning models used by the trading system.

These models are organized around individual Forex symbols.

The system is therefore capable of loading a model corresponding to the market currently being analyzed.

Conceptually:

```text
EURUSD
   │
   └── EURUSD Model

GBPUSD
   │
   └── GBPUSD Model

USDJPY
   │
   └── USDJPY Model

AUDUSD
   │
   └── AUDUSD Model

...
```

This allows each market to have its own learned behavior rather than forcing one model to treat every Forex pair identically.

---

# 📦 Model Usage

A simplified version of the model-loading architecture looks like:

```python
import joblib

model = joblib.load("path/to/model.pkl")

prediction = model.predict(features)
```

The actual repository contains the project's complete model-loading and prediction workflow.

The model artifacts may include trained models, serialized objects, and other information required by the trading engine.

---

# 📊 2. CSV FILES

The `CSV FILES` directory contains market datasets collected from the MT5 environment connected to Exness.

These datasets are used for:

- Model training
- Feature engineering
- Historical analysis
- Threshold selection
- Strategy testing
- Forward testing
- Model validation
- Market research
- Live-data preparation

The datasets are organized around the supported Forex symbols.

---

# 📈 Market Data Pipeline

The general data pipeline is:

```text
Exness
   │
   ▼
MetaTrader 5
   │
   ▼
Python MT5 API
   │
   ▼
OHLC / Market Data
   │
   ▼
CSV Dataset
   │
   ▼
Feature Engineering
   │
   ▼
Machine Learning
```

The repository therefore provides both the **software required to retrieve data** and the datasets used during development.

---

# 🧮 3. PY FILES

The `PY FILES` directory contains the Python source code responsible for the complete trading workflow.

The scripts cover:

- MT5 connection
- Dataset collection
- Feature engineering
- Model training
- Model inspection
- Threshold optimization
- Forward testing
- Realistic forward testing
- Live trading
- Risk management
- Trade execution
- Position management
- Logging
- Symbol configuration

---

# 🛠️ Python Files Explained

## `func.py`

`func.py` contains shared helper functions used throughout the project.

Instead of duplicating the same functionality across many scripts, common operations are centralized in this module.

Examples include:

```python
from func import (
    calc_lot_size,
    place_buy,
    place_sell,
    rollover_sleep,
    move_sl_and_partial_close,
    get_pip_info,
    forex_market_open
)
```

Other helper functionality includes:

```python
from func import (
    log_trade,
    Entry_Filtering,
    Data_Prediction,
    set_logger,
    correlation_check,
    spread_filter,
    normalize_symbol
)
```

This makes `func.py` one of the core utility modules of the project.

---

# 💰 Risk Management Functions

The project includes dedicated functionality for managing trade risk.

Examples include:

```python
calc_lot_size()
```

This function is responsible for calculating an appropriate trading volume based on the configured risk-management logic.

Other components handle:

- Stop-loss management
- Take-profit management
- Position modification
- Partial closing
- Pip calculations
- Symbol specifications
- Trade logging

The objective is to ensure that trade execution is not simply based on a prediction but also passes through the appropriate risk-management layer.

---

# 📉 ATR-Based Trading Architecture

A significant part of the system is built around **Average True Range (ATR)**.

ATR is used to measure market volatility and can be incorporated into:

- Stop-loss calculation
- Take-profit calculation
- Trade sizing
- Market movement analysis
- Threshold analysis
- Trade simulation
- Strategy testing

The repository contains several scripts dedicated to ATR-based research and execution.

---

# 🔬 `ATR_TRAIN_MODEL.py`

This script is responsible for the machine learning model training workflow associated with the ATR-based strategy.

The general workflow is:

```text
Historical Data
      │
      ▼
Feature Engineering
      │
      ▼
ATR / Market Features
      │
      ▼
Training Dataset
      │
      ▼
Machine Learning Model
      │
      ▼
Trained Model
```

---

# 🎯 `ATR_FIND_THRESHOLD.py`

This component is used to investigate appropriate prediction thresholds for the trading strategy.

Instead of blindly entering every time the model produces a prediction, the system can evaluate different confidence thresholds and determine how trading performance changes.

Conceptually:

```text
Model Probability
       │
       ▼
Threshold Analysis
       │
       ├── Trade Count
       ├── Win Rate
       ├── Profit Factor
       ├── Drawdown
       ├── Expectancy
       └── Other Metrics
       │
       ▼
Selected Threshold
```

---

# 🔎 `ATR_ALL_FIND_THRESHOLD.py`

This script extends threshold analysis across multiple symbols.

Instead of manually evaluating one market at a time, the system can perform threshold research across the supported trading universe.

This makes it possible to compare model behavior between different Forex pairs.

---

# 🧪 `ATR_ANALYSIS_&_TEST.py`

This script is used for analysis and testing of the trading strategy.

It can be used to investigate:

- Model predictions
- Trade outcomes
- Strategy behavior
- Market conditions
- Threshold performance
- Trading statistics
- Historical performance

---

# 🔄 `ATR_FORWARD_TEST_MODEL.py`

Forward testing is an important part of the development workflow.

Instead of training and evaluating a model exclusively on the same historical data, forward testing evaluates how the strategy behaves on data that was not used for the original model development process.

Conceptually:

```text
Historical Data
      │
      ├──────────────► Training
      │
      └──────────────► Model Development
                         │
                         ▼
                    New Data
                         │
                         ▼
                   Forward Test
                         │
                         ▼
                  Performance
```

---

# 🧪 `ATR_REALISTIC_FOWARDT.py`

This script is designed for more realistic forward-testing scenarios.

Realistic testing is particularly important for automated trading because live execution introduces conditions that simplified backtests may not fully represent.

These can include:

- Spread
- Execution conditions
- Trade timing
- Market sessions
- Position management
- Entry filtering
- Broker-specific symbol behavior
- Slippage and execution differences

---

# 🤖 `ATR_TRADE_HIGH_NXT.py`

This module contains the core trading logic used by the live trading architecture.

The live trading process can be summarized as:

```text
MT5 Market Data
      │
      ▼
Feature Engineering
      │
      ▼
Model Prediction
      │
      ▼
Signal Evaluation
      │
      ▼
Entry Filtering
      │
      ├── Spread Check
      ├── Correlation Check
      ├── Market Open Check
      ├── Signal Conditions
      └── Other Filters
      │
      ▼
Risk Calculation
      │
      ▼
Order Execution
      │
      ▼
Trade Management
```

---

# 🔌 `CONNECT_MT5.py`

This module is responsible for connecting the Python environment to MetaTrader 5.

The MT5 connection provides access to:

- Account information
- Symbol information
- Market prices
- Historical rates
- Current market data
- Trading functions
- Position information
- Order execution

The exact MT5 configuration depends on the user's local installation and broker environment.

---

# 📥 `GET_MT5_DATASET.py`

This script is responsible for collecting market data from MT5 and storing it for further analysis.

The resulting data can then be used by:

- Training scripts
- Threshold scripts
- Forward-testing scripts
- Analysis scripts
- Strategy development

---

# 📋 `LOAD_MODEL_INFOS.py`

This utility is used to inspect information associated with trained models.

It can be useful when working with a large number of serialized models and trying to determine:

- Which symbol a model belongs to
- Model configuration
- Model metadata
- Saved model information
- Model-related parameters

---

# 📊 `SYMBOL_DATA.py`

This module contains symbol-specific configuration and/or data-related logic.

It provides a central location for information required when processing different Forex instruments.

---

# 📐 `SYMBOL_POINTS.py`

This module contains symbol point-related configuration.

Different Forex instruments can have different pricing precision and point/pip behavior.

Centralizing this information helps the trading engine correctly interpret:

- Price movements
- Pip distances
- Stop-loss distances
- Take-profit distances
- Position calculations

---

# ⚙️ Live Multi-Process Trading

One of the important features of the architecture is the ability to run multiple symbols simultaneously.

The main trading launcher uses Python's `multiprocessing` functionality.

Example:

```python
from multiprocessing import Process
from ATR_TRADE_HIGH_NXT import FOREX_TRADING

import logging


def setup_logger(symbol):

    logger = logging.getLogger(symbol)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(
            f'\n %(asctime)s | [{symbol}] | %(message)s \n'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def run_symbol(symbol):

    logger = setup_logger(symbol)

    logger.info(
        f'\n CURRENTLY RUNNING THIS SYMBOL [[ {symbol} ]] \n'
    )

    FOREX_TRADING(
        SYMBOL=symbol,
        logger=logger
    )


if __name__ == "__main__":

    symbols = [
        "AUDCAD",
        "AUDUSD",
        "CADJPY",
        "EURJPY",
        "EURUSD",
        "GBPJPY",
        "GBPUSD",
        "NZDJPY",
        "NZDUSD",
        "USDCHF",
        "USDJPY"
    ]

    symbols = [symbol + 'c' for symbol in symbols]

    processes = []

    for symbol in symbols:

        p = Process(
            target=run_symbol,
            args=(symbol,)
        )

        p.start()

        processes.append(p)

    for p in processes:

        p.join()
```

This architecture allows each symbol to run inside its own process.

Conceptually:

```text
                    MAIN PROCESS
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    AUDCADc          AUDUSDc          CADJPYc
        │                │                │
        ▼                ▼                ▼
    Process 1         Process 2         Process 3
        │                │                │
        ▼                ▼                ▼
      Model            Model            Model
        │                │                │
        ▼                ▼                ▼
     Trading          Trading          Trading
```

Additional symbols can be added to the configuration as required.

---

# 🧩 Entry Filtering

The trading engine does not necessarily treat every machine learning prediction as an automatic trade.

The project contains an `Entry_Filtering()` component designed to provide an additional layer between model prediction and execution.

Conceptually:

```text
Machine Learning Prediction
            │
            ▼
      Entry Filtering
            │
       ┌────┴────┐
       │         │
    PASS       REJECT
       │         │
       ▼         ▼
    Trade      No Trade
```

This architecture is useful because a machine learning model can produce a prediction while the broader market conditions may still be unsuitable for entering a position.

---

# 📏 Spread Filtering

The system contains a `spread_filter()` component.

Spread is an important consideration in automated Forex trading because transaction costs can significantly affect short-term strategies.

The trading system can therefore evaluate spread conditions before entering a position.

```text
Current Spread
      │
      ▼
Spread Filter
      │
   ┌──┴──┐
   │     │
 PASS  REJECT
   │     │
   ▼     ▼
Trade  No Trade
```

---

# 🔗 Correlation Checking

The system also contains a `correlation_check()` component.

Correlation analysis can be useful when multiple currency pairs are being traded simultaneously.

For example, several currency pairs may have relationships that cause multiple positions to effectively represent similar market exposure.

A correlation layer can therefore help evaluate portfolio-level trade conditions rather than considering each trade completely independently.

---

# 🌍 Forex Market Status

The system contains:

```python
forex_market_open()
```

This allows the trading engine to determine whether the Forex market is currently available for trading before proceeding with execution.

---

# 💤 Rollover Handling

The system contains:

```python
rollover_sleep()
```

This is designed to handle periods around the daily Forex rollover where trading conditions may differ from normal market conditions.

---

# 📦 Position Sizing

Position sizing is handled through:

```python
calc_lot_size()
```

The goal is to determine an appropriate trade volume based on the strategy's configured risk-management rules.

This is important because a trading signal alone does not determine how much capital should be exposed.

The architecture is therefore:

```text
Trading Signal
      │
      ▼
Risk Parameters
      │
      ▼
Account Information
      │
      ▼
Lot Size Calculation
      │
      ▼
Order Volume
```

---

# 🛑 Stop Loss & Take Profit

Trade management includes functionality for managing protective and profit-taking levels.

The project contains functions such as:

```python
place_buy()
place_sell()
move_sl_and_partial_close()
```

These functions form part of the automated execution and position-management layer.

---

# 📒 Trade Logging

The system includes:

```python
log_trade()
```

Trade logging is important for analyzing what the algorithm actually did during execution.

A trading log can be used to investigate:

- Entry time
- Exit time
- Symbol
- Direction
- Trade result
- Model signal
- Trade conditions
- Execution behavior
- Position management

This creates a feedback loop between live execution and later analysis.

---

# 🔄 Symbol Normalization

The project includes:

```python
normalize_symbol()
```

This is particularly useful when working with broker-specific symbol naming conventions.

For example, a broker may use:

```text
EURUSDc
```

while research code may refer to:

```text
EURUSD
```

A normalization layer helps keep the internal strategy logic consistent.

---

# 🧠 Data Prediction Pipeline

The repository contains a centralized:

```python
Data_Prediction()
```

component.

The conceptual process is:

```text
MT5 Data
   │
   ▼
Feature Processing
   │
   ▼
Model Input
   │
   ▼
Trained Model
   │
   ▼
Prediction
   │
   ▼
Trading Decision
```

---

# 🔬 Development Pipeline

The complete development workflow can be represented as:

```text
                    HISTORICAL MARKET DATA
                              │
                              ▼
                       FEATURE ENGINEERING
                              │
                              ▼
                         MODEL TRAINING
                              │
                              ▼
                       MODEL EVALUATION
                              │
                              ▼
                     THRESHOLD OPTIMIZATION
                              │
                              ▼
                       FORWARD TESTING
                              │
                              ▼
                  REALISTIC FORWARD TESTING
                              │
                              ▼
                         LIVE TESTING
                              │
                              ▼
                       LIVE EXECUTION
                              │
                              ▼
                       TRADE LOGGING
                              │
                              ▼
                         PERFORMANCE
                           ANALYSIS
```

---

# 🧪 Backtesting vs Forward Testing

The project does not rely exclusively on historical model performance.

It includes separate tools for forward testing.

This distinction is important.

### Historical Testing

The strategy is evaluated using historical market data.

```text
Historical Data
       │
       ▼
Strategy
       │
       ▼
Historical Results
```

### Forward Testing

The model/strategy is evaluated on data that represents a later period.

```text
Training Data
       │
       ▼
Model
       │
       ▼
Previously Unseen Data
       │
       ▼
Forward Results
```

### Realistic Forward Testing

Additional execution-related considerations are incorporated into the evaluation process.

```text
Unseen Market Data
       │
       ▼
Trading Strategy
       │
       ├── Spread
       ├── Entry Conditions
       ├── Position Management
       ├── Timing
       └── Execution Conditions
       │
       ▼
Realistic Results
```

---

# 📊 Why Multiple Models?

Financial markets are not identical.

Different currency pairs can exhibit different:

- Volatility
- Liquidity
- Trading sessions
- Price behavior
- Spread characteristics
- Correlations
- Market reactions

For this reason, the architecture uses symbol-specific models rather than assuming that one universal model must perform identically across every market.

---

# 🏗️ System Architecture

The overall system can be visualized as:

```text
                    ┌─────────────────────┐
                    │      EXNESS         │
                    │     BROKER          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   MetaTrader 5      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Python / MT5     │
                    │    Data Interface   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Market Dataset    │
                    │      CSV Files      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Machine Learning    │
                    │      Models         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Prediction /      │
                    │ Threshold Analysis  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Entry Filtering    │
                    ├─────────────────────┤
                    │ Spread Filter       │
                    │ Correlation Check   │
                    │ Market Status       │
                    │ Other Conditions    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Management   │
                    ├─────────────────────┤
                    │ Lot Size            │
                    │ Stop Loss           │
                    │ Take Profit         │
                    │ Position Management │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   MT5 Execution     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Trade Logging    │
                    └─────────────────────┘
```

---

# 💻 Technology Stack

The project is primarily built using:

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| MetaTrader 5 | Market data and trade execution interface |
| Exness | Broker environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Joblib | Model serialization/loading |
| Machine Learning | Market prediction |
| CSV | Dataset storage |
| Multiprocessing | Parallel multi-symbol execution |
| Logging | Runtime and trade monitoring |

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
```

Enter the project directory:

```bash
cd <YOUR-REPOSITORY-DIRECTORY>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install MetaTrader5 pandas numpy joblib
```

Additional dependencies may be required depending on which research or analysis scripts you run.

---

# 🖥️ MetaTrader 5 Setup

Before running the live trading components:

1. Install MetaTrader 5.
2. Log in to the appropriate trading account.
3. Make sure the required symbols are available in MT5.
4. Verify that the symbols use the expected broker naming convention.
5. Confirm that algorithmic trading is enabled where required.
6. Confirm that Python can communicate with the MT5 terminal.
7. Test the connection using the MT5 connection script.

The live system should never be connected to a live account until the complete execution workflow has been tested.

---

# 🔐 Credentials & Security

**Never upload sensitive credentials to GitHub.**

Do not commit:

```text
MT5 account number
MT5 password
Broker server credentials
API keys
Private tokens
Personal account information
```

Use environment variables or a local configuration file excluded through `.gitignore`.

Example:

```text
.env
credentials.json
secrets.json
```

Recommended `.gitignore` entries:

```gitignore
.env
*.env
credentials.json
secrets.json
__pycache__/
*.pyc
```

---

# ▶️ Running the Trading System

After configuring MT5 and the required models, the multi-symbol trading launcher can be started with:

```bash
python FOREX_TRADING.py
```

The launcher creates a separate process for each configured symbol.

Example:

```text
CURRENTLY RUNNING THIS SYMBOL [[ AUDCADc ]]

CURRENTLY RUNNING THIS SYMBOL [[ AUDUSDc ]]

CURRENTLY RUNNING THIS SYMBOL [[ CADJPYc ]]

CURRENTLY RUNNING THIS SYMBOL [[ EURJPYc ]]

CURRENTLY RUNNING THIS SYMBOL [[ EURUSDc ]]

CURRENTLY RUNNING THIS SYMBOL [[ GBPJPYc ]]

CURRENTLY RUNNING THIS SYMBOL [[ GBPUSDc ]]

CURRENTLY RUNNING THIS SYMBOL [[ NZDJPYc ]]

CURRENTLY RUNNING THIS SYMBOL [[ NZDUSDc ]]

CURRENTLY RUNNING THIS SYMBOL [[ USDCHFc ]]

CURRENTLY RUNNING THIS SYMBOL [[ USDJPYc ]]
```

Each process runs its own trading workflow.

---

# 🧰 Research Workflow

If you are developing or modifying the strategy, the recommended workflow is:

### Step 1 — Collect Data

Run:

```text
GET_MT5_DATASET.py
```

---

### Step 2 — Train Models

Run:

```text
ATR_TRAIN_MODEL.py
```

---

### Step 3 — Inspect Models

Run:

```text
LOAD_MODEL_INFOS.py
```

---

### Step 4 — Analyze Strategy

Run:

```text
ATR_ANALYSIS_&_TEST.py
```

---

### Step 5 — Find Trading Thresholds

Run:

```text
ATR_FIND_THRESHOLD.py
```

or:

```text
ATR_ALL_FIND_THRESHOLD.py
```

---

### Step 6 — Forward Test

Run:

```text
ATR_FORWARD_TEST_MODEL.py
```

---

### Step 7 — Realistic Forward Test

Run:

```text
ATR_REALISTIC_FOWARDT.py
```

---

### Step 8 — Live Execution

Only after extensive testing:

```text
FOREX_TRADING.py
```

---

# 📁 Detailed File Reference

| File | Purpose |
|---|---|
| `ATR_ALL_FIND_THRESHOLD.py` | Multi-symbol threshold analysis |
| `ATR_ANALYSIS_&_TEST.py` | Strategy analysis and testing |
| `ATR_FIND_THRESHOLD.py` | Prediction threshold research |
| `ATR_FORWARD_TEST_MODEL.py` | Forward-testing workflow |
| `ATR_REALISTIC_FOWARDT.py` | Realistic forward testing |
| `ATR_TRADE_HIGH_NXT.py` | Core ATR-based trading engine |
| `ATR_TRAIN_MODEL.py` | Model training |
| `CONNECT_MT5.py` | MT5 connection |
| `FOREX_TRADING.py` | Multi-symbol trading launcher |
| `func.py` | Shared helper functions |
| `GET_MT5_DATASET.py` | MT5 dataset collection |
| `LOAD_MODEL_INFOS.py` | Model inspection |
| `SYMBOL_DATA.py` | Symbol/data configuration |
| `SYMBOL_POINTS.py` | Symbol point/pip configuration |

---

# 🔁 End-to-End Example

A simplified representation of the complete system is:

```text
                    START
                      │
                      ▼
                Connect to MT5
                      │
                      ▼
               Select Symbol
                      │
                      ▼
             Retrieve Market Data
                      │
                      ▼
             Prepare Features
                      │
                      ▼
              Load ML Model
                      │
                      ▼
              Generate Prediction
                      │
                      ▼
             Check Threshold
                      │
                ┌─────┴─────┐
                │           │
             REJECT        PASS
                │           │
                ▼           ▼
              WAIT     Entry Filtering
                            │
                            ▼
                      Spread Filter
                            │
                            ▼
                    Correlation Check
                            │
                            ▼
                    Market Open Check
                            │
                       ┌────┴────┐
                       │         │
                    REJECT      PASS
                       │         │
                       ▼         ▼
                     WAIT   Calculate Risk
                                  │
                                  ▼
                             Calculate Lot
                                  │
                                  ▼
                           Place BUY / SELL
                                  │
                                  ▼
                           Manage Position
                                  │
                                  ▼
                         SL / TP / Partial
                                  │
                                  ▼
                           Log Trade
                                  │
                                  ▼
                              Repeat
```

---

# 📈 Performance Evaluation

The repository is designed to allow strategy performance to be evaluated using multiple metrics rather than relying on a single number.

Useful metrics include:

- Number of trades
- Win rate
- Profit factor
- Maximum drawdown
- Expectancy
- Sharpe ratio
- Average trade
- Average win
- Average loss
- Risk/reward
- Trade frequency
- Symbol-specific performance
- Threshold-specific performance

A robust evaluation should consider multiple metrics simultaneously.

For example:

```text
High Win Rate
       +
Low Drawdown
       +
Positive Expectancy
       +
Acceptable Trade Frequency
       +
Stable Forward-Test Performance
       =
More Informative Strategy Evaluation
```

---

# 🧪 Why Forward Testing Matters

A machine learning trading system can perform very well on historical data while behaving differently on new market data.

This can happen because of:

- Market regime changes
- Overfitting
- Data leakage
- Changing volatility
- Spread changes
- Liquidity changes
- News events
- Different market sessions
- Execution differences

For this reason, this project includes dedicated forward-testing components rather than relying exclusively on training performance.

---

# 🧠 Machine Learning Philosophy

The goal of this project is not simply:

```text
MODEL → BUY/SELL
```

Instead, the architecture attempts to create a layered decision-making system:

```text
MODEL
  │
  ▼
Prediction
  │
  ▼
Confidence / Threshold
  │
  ▼
Market Conditions
  │
  ▼
Entry Filters
  │
  ▼
Risk Management
  │
  ▼
Execution
  │
  ▼
Position Management
```

This separation makes the system easier to analyze, test, and improve.

---

# 🧩 Modular Architecture

One of the main design principles of the project is modularity.

Instead of placing the entire trading system inside one massive script, functionality is separated into different modules.

For example:

```text
CONNECT_MT5.py
       │
       ▼
GET_MT5_DATASET.py
       │
       ▼
ATR_TRAIN_MODEL.py
       │
       ▼
ATR_FIND_THRESHOLD.py
       │
       ▼
ATR_FORWARD_TEST_MODEL.py
       │
       ▼
ATR_REALISTIC_FOWARDT.py
       │
       ▼
ATR_TRADE_HIGH_NXT.py
       │
       ▼
FOREX_TRADING.py
```

This makes individual parts easier to test and modify.

---

# 🔧 Customization

The system can be adapted by changing:

- Supported symbols
- Model files
- Trading thresholds
- ATR parameters
- Risk parameters
- Entry filters
- Spread conditions
- Correlation conditions
- Trading sessions
- Position-management rules
- Dataset periods
- Feature engineering
- Model architecture

---

# ➕ Adding a New Symbol

A new symbol can be added to the symbol configuration.

Example:

```python
symbols = [
    "AUDCAD",
    "AUDUSD",
    "CADJPY",
    "EURJPY",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "NZDJPY",
    "NZDUSD",
    "USDCHF",
    "USDJPY",
    "USDCAD"
]
```

If the broker requires a suffix:

```python
symbols = [symbol + 'c' for symbol in symbols]
```

However, adding a symbol also requires the corresponding:

- Dataset
- Model
- Symbol configuration
- Point/pip configuration
- Validation
- Testing

---

# 🧪 Testing Philosophy

Before enabling live execution, the strategy should pass through several stages:

```text
Code Testing
     │
     ▼
Historical Analysis
     │
     ▼
Model Validation
     │
     ▼
Threshold Testing
     │
     ▼
Forward Testing
     │
     ▼
Realistic Forward Testing
     │
     ▼
Demo Testing
     │
     ▼
Small-Scale Live Testing
     │
     ▼
Full Deployment
```

Skipping these stages can introduce unnecessary risk.

---

# 📌 Project Goals

The long-term goal of this project is to build a complete automated trading infrastructure capable of:

- Collecting real market data
- Training machine learning models
- Evaluating trading signals
- Selecting appropriate thresholds
- Testing strategies on unseen data
- Filtering unsuitable trades
- Managing portfolio exposure
- Calculating position sizes
- Executing trades automatically
- Managing open positions
- Recording trade activity
- Continuously analyzing performance

---

# 🛡️ Risk Management Comes First

A machine learning prediction is only one component of the system.

The project treats risk management as an independent layer.

```text
Prediction ≠ Trade
```

Instead:

```text
Prediction
    +
Threshold
    +
Market Conditions
    +
Entry Filters
    +
Spread
    +
Correlation
    +
Risk Management
    +
Execution Conditions
    =
Trade Decision
```

This distinction is important when developing automated trading systems.

---

# 🌐 Supported Trading Environment

The project was developed around:

```text
Python
MetaTrader 5
Exness
Forex Markets
Machine Learning
Automated Execution
```

The exact behavior of the system can depend on:

- Broker
- Account type
- Symbol suffixes
- Market conditions
- MT5 terminal configuration
- Available historical data
- Execution environment

---

# 📚 Educational Purpose

This repository can also serve as a practical example of how different areas of software engineering can be combined into one larger system:

```text
Python Programming
        +
Data Engineering
        +
Machine Learning
        +
Financial Data Analysis
        +
Algorithmic Trading
        +
Risk Management
        +
Automation
        +
Software Architecture
```

It demonstrates how a machine learning model can be integrated into a larger real-world application rather than being used as an isolated prediction script.

---

# 🚧 Future Improvements

Potential future improvements include:

- Better experiment tracking
- Automated model retraining
- More sophisticated portfolio management
- Improved execution monitoring
- Persistent database-based trade storage
- Real-time dashboards
- Advanced risk controls
- Automated model comparison
- Model drift detection
- Automated performance reports
- More comprehensive walk-forward testing
- Improved failure recovery
- Better configuration management
- Docker/container deployment
- Cloud-based monitoring
- More extensive multi-timeframe analysis

---

# 🤝 Contributing

Contributions, ideas, bug reports, and improvements are welcome.

If you want to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test your changes thoroughly.
5. Submit a pull request.

For significant strategy changes, include an explanation of:

- What was changed
- Why it was changed
- How it was tested
- What effect it had on the strategy

---

# ⚠️ Important Security Reminder

Do **NOT** commit your:

```text
MT5 Login
MT5 Password
Broker Password
API Keys
Private Credentials
Personal Account Information
```

into this repository.

If credentials have ever been accidentally committed, rotate/revoke them immediately.

---

# 👨‍💻 Author

## Ezee Kits

Electrical & Electronics Engineer | Python Developer | Automation | Machine Learning | Algorithmic Trading

The project combines interests in:

- Python
- Machine Learning
- Artificial Intelligence
- Data Engineering
- Automation
- Financial Technology
- Algorithmic Trading
- Software Development
- 
Portfolio:
https://ezee-kits-portfolio.onrender.com/

GitHub:
https://github.com/Ezee-Kits/

YouTube:
https://www.youtube.com/@EzeeKits
---

# ⭐ Support the Project

If you find this project useful for learning about:

- Machine learning
- Python automation
- MT5 integration
- Algorithmic trading
- Data engineering
- Quantitative research

consider giving the repository a ⭐.

Your support helps encourage further development and documentation.

---

# 📜 License

This repository is intended primarily for educational and research purposes.

If a formal license file is included in the repository, refer to that license for the applicable terms.

---

# 🚀 Final Architecture Summary

The project can ultimately be summarized as:

```text
                         ┌─────────────────────┐
                         │       EXNESS        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   MetaTrader 5      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Market Data      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Machine Learning    │
                         │      Models         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Prediction /        │
                         │ Threshold Analysis  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Entry Filtering     │
                         └──────────┬──────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      │             │             │
                      ▼             ▼             ▼
                   Spread      Correlation    Market
                   Filter         Check        Status
                      │             │             │
                      └─────────────┼─────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Risk Management    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Position Sizing     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Trade Execution     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Position Management │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Trade Logging     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Performance Analysis│
                         └─────────────────────┘
```

---

# 🔥 Project in One Sentence

**An end-to-end Python machine-learning Forex trading system that connects MetaTrader 5 to an Exness trading environment, processes multi-symbol market data, uses symbol-specific trained models for prediction, validates signals through multiple trading filters, applies risk management and ATR-based trade logic, and supports automated multi-symbol trade execution and position management.**
