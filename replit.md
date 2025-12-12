# Blueprint Trader AI

## Overview
Blueprint Trader AI is an automated trading bot designed for FTMO Challenge accounts. It operates 24/7 on a Windows VM with MetaTrader 5, executing a pre-validated trading strategy. A Flask web server offers a lightweight monitoring interface, while all trading operations are independently managed on the VM. The project aims to automate the process of passing FTMO challenges, leveraging a robust strategy and sophisticated risk management to achieve consistent profitability within FTMO's stringent rules.

## User Preferences
- Preferred communication style: Simple, everyday language
- Strategy must use EXACT SAME logic as backtests
- Bot must trade independently (no Discord dependency for trades)
- Pre-trade risk checks to prevent FTMO rule violations
- Using FTMO demo account for trading
- Using OANDA API for data fetching

## System Architecture
The system is composed of two primary components:
1.  **Standalone MT5 Bot (`main_live_bot.py`)**: Runs continuously on a Windows VM, responsible for all trading activities. It directly utilizes `strategy_core.py` for signal generation, includes pre-trade risk simulation to prevent drawdown breaches, features auto-reconnection for MT5, and is configured for auto-start on boot.
2.  **Minimal Flask Web Server (`main.py`)**: Provides a lightweight monitoring interface.

The core trading strategy is built upon "7 Confluence Pillars" evaluating setups based on HTF Bias, Location, Fibonacci, Liquidity, Structure, Confirmation, and Risk-to-Reward (R:R). Trade signals are classified as `ACTIVE` (execute trade), `WATCHING` (monitor), or `SCAN` (no action) based on confluence and quality.

A 7-layer safety system, known as "Challenge Mode Elite Protection," is implemented to maximize FTMO challenge pass rates. This includes:
-   **Global Risk Controller**: Real-time P/L tracking and proactive SL adjustment.
-   **Dynamic Position Sizing**: Adaptive sizing based on drawdown and confluence.
-   **Smart Concurrent Trade Limit**: Limits open and pending positions.
-   **Pending Order Management**: Risk and time-based cancellation.
-   **Live Equity Protection Loop**: 30-second monitoring with automatic protective actions.
-   **Challenge-Optimized Behavior**: Five risk modes (Aggressive to Halted) based on current drawdown.
-   **Core Strategy Integration**: Wraps the core strategy with safety layers without altering entry logic.

The system supports 34 tradable assets, including major and cross Forex pairs, Gold (XAUUSD), Silver (XAGUSD), Bitcoin (BTCUSD), Ethereum (ETHUSD), S&P 500 (US500), and Nasdaq 100 (US100). A `symbol_mapping.py` module handles conversions between OANDA data formats and FTMO MT5 trading formats.

Key architectural decisions include:
-   Modular package structure under `/tradr/` for strategy, risk, MT5 integration, data handling, and utilities.
-   Use of pending limit orders for precise trade entry matching backtest conditions.
-   Comprehensive FTMO challenge rule enforcement, including profit targets, max daily loss, and max total drawdown.
-   Dynamic lot sizing and H4 timeframe for Stop Loss calculation for optimized performance.

## External Dependencies
-   **MetaTrader5**: For trading operations (Windows VM only).
-   **OANDA API**: For fetching market data.
-   **Flask**: For the lightweight monitoring web server.
-   **pandas**: For data manipulation.
-   **numpy**: For numerical operations.
-   **requests**: For HTTP requests.
-   **python-dotenv**: For environment variable management.