# ===============================================================================================
# FORWARD TEST + MODEL OPPORTUNITY TIME ANALYSIS
# EURUSD 10M
# ===============================================================================================

import json
import os
import joblib
import numpy as np
import pandas as pd

from func import apply_features, normalize_symbol


# ===============================================================================================
# SETTINGS
# ===============================================================================================

SYMBOL = "EURUSD"

HIGH_TARGETS = ['THL_3H']

CD_TIME = '10M'

weights = {
    'THL_3H': 1.0
}

risk_percent = 1

W_threshold = 0.503

tp_mult = 1.75
sl_mult = 1.75

INITIAL_BALANCE = 1201

BASE_PATH = "/content/drive/MyDrive/FOREX TRADING/ALL_MODELS"

DATA_PATH = (
    "/content/drive/MyDrive/FOREX TRADING/CSV_FILES/"
    "MT5_10M_BT_EURUSD_Exchange_Rate_Dataset.csv"
)

SYMBOL_INFO_PATH = (
    "/content/drive/MyDrive/FOREX TRADING/CSV_FILES/"
    "SYMBOL_INFO.json"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/FOREX TRADING/"
    "FORWARD_TEST_ANALYSIS"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================================================================================
# LOAD MODELS
# ===============================================================================================

def load_models(model_type, CD_TIME, symbol):

    symbol = normalize_symbol(symbol)

    models_dict = {}

    for target in HIGH_TARGETS:

        file_path = (
            f"{BASE_PATH}/"
            f"HL_{model_type}_{target}_{CD_TIME}_{symbol}_model.pkl"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model not found:\n{file_path}"
            )

        bundle = joblib.load(file_path)

        models_dict[target] = {
            "model": bundle["model"],
            "features": bundle["features"]
        }

        print(
            f"[[GOOD]] Loaded {model_type} model for {target}"
        )

    return models_dict


HELPER_MODELS = load_models(
    "MAIN",
    CD_TIME,
    SYMBOL
)


# ===============================================================================================
# LOAD SYMBOL INFORMATION
# ===============================================================================================

with open(SYMBOL_INFO_PATH, "r") as f:
    SYMBOL_INFO = json.load(f)

info = SYMBOL_INFO[SYMBOL]

pip_size = info["pip_size"]

pip_value_per_lot = info["pip_value_per_lot"]

vol_info = info["vol_info"]


# ===============================================================================================
# LOAD FORWARD-TEST DATA
# ===============================================================================================

print("\n============================================================")
print("LOADING FORWARD TEST DATA")
print("============================================================")

data = pd.read_csv(DATA_PATH)

print(f"Rows loaded : {len(data):,}")

print(f"Columns     : {list(data.columns)}")


# ===============================================================================================
# CHECK REQUIRED COLUMNS
# ===============================================================================================

required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Spread"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ===============================================================================================
# DATE
# ===============================================================================================

data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce"
)

data = data.dropna(
    subset=["Date"]
)

data = data.sort_values(
    "Date"
).reset_index(drop=True)


print(
    f"\nSTART DATE : {data['Date'].iloc[0]}"
)

print(
    f"END DATE   : {data['Date'].iloc[-1]}"
)


# ===============================================================================================
# APPLY FEATURES
# ===============================================================================================

print("\n============================================================")
print("APPLYING FEATURES")
print("============================================================")

df = apply_features(data.copy())

df.dropna(
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

print(
    f"Feature dataset rows : {len(df):,}"
)


# ===============================================================================================
# PRESERVE DATE + HISTORICAL SPREAD
# ===============================================================================================

# Make sure Date and Spread survived apply_features()

if "Date" not in df.columns:

    raise ValueError(
        "Date column was lost inside apply_features(). "
        "Modify apply_features() so that Date is preserved."
    )

if "Spread" not in df.columns:

    raise ValueError(
        "Spread column was lost inside apply_features(). "
        "Modify apply_features() so that Spread is preserved."
    )


# ===============================================================================================
# WORKING DATAFRAME
# ===============================================================================================

working_df = df[
    [
        "Date",
        "High",
        "Low",
        "Open",
        "Close",
        "ATR",
        "Spread"
    ]
].copy()


# ===============================================================================================
# MODEL PREDICTIONS
# ===============================================================================================

print("\n============================================================")
print("GENERATING MODEL PREDICTIONS")
print("============================================================")


for target in HIGH_TARGETS:

    print(
        f"\n[[ CURRENTLY PREDICTING : {target} ]]"
    )

    help_model = HELPER_MODELS[target]["model"]

    help_cols = HELPER_MODELS[target]["features"]

    # Verify features exist
    missing_features = [
        col for col in help_cols
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing model features for {target}: "
            f"{missing_features}"
        )

    X_help = df[help_cols]

    help_proba = help_model.predict_proba(
        X_help
    )

    up_prob = help_proba[:, 1]

    down_prob = help_proba[:, 0]

    working_df[
        f"{target}_UP"
    ] = up_prob

    working_df[
        f"{target}_DN"
    ] = down_prob


# ===============================================================================================
# WEIGHTED PREDICTION
# ===============================================================================================

weight_list = [
    weights[t]
    for t in HIGH_TARGETS
]

total_weight = sum(weight_list)


working_df["UP_AVG"] = (
    sum(
        working_df[f"{t}_UP"] * weights[t]
        for t in HIGH_TARGETS
    )
    / total_weight
)


working_df["DN_AVG"] = (
    sum(
        working_df[f"{t}_DN"] * weights[t]
        for t in HIGH_TARGETS
    )
    / total_weight
)


# ===============================================================================================
# ATR / TP / SL
# ===============================================================================================

working_df["ATR_PIPS"] = (
    working_df["ATR"] / pip_size
)

working_df["SL_PIPS"] = (
    working_df["ATR_PIPS"] * sl_mult
)

working_df["TP_PIPS"] = (
    working_df["ATR_PIPS"] * tp_mult
)


# ===============================================================================================
# SIGNAL
# ===============================================================================================

working_df["SIGNAL"] = 0

working_df.loc[
    working_df["UP_AVG"] >= W_threshold,
    "SIGNAL"
] = 1

working_df.loc[
    working_df["DN_AVG"] >= W_threshold,
    "SIGNAL"
] = -1


# ===============================================================================================
# NEXT CANDLE ENTRY
# ===============================================================================================

working_df["NEXT_ENTRY"] = (
    working_df["Open"].shift(-1)
)

working_df["ENTRY_DATE"] = (
    working_df["Date"].shift(-1)
)


# ===============================================================================================
# IMPORTANT:
# HISTORICAL SPREAD
#
# Spread values are assumed to be MT5 POINTS.
#
# Example:
# EURUSD 5-digit broker:
#
# 7 points = 0.7 pip
#
# If your dataset's Spread is already in pips,
# change SPREAD_IS_POINTS to False.
# ===============================================================================================

SPREAD_IS_POINTS = True


if SPREAD_IS_POINTS:

    working_df["SPREAD_PIPS"] = (
        working_df["Spread"]
        * (0.00001 / pip_size)
    )

else:

    working_df["SPREAD_PIPS"] = (
        working_df["Spread"]
    )


# ===============================================================================================
# ENTRY PRICES
# ===============================================================================================

working_df["ASK_PRICE"] = (
    working_df["NEXT_ENTRY"]
    + (
        working_df["SPREAD_PIPS"]
        * pip_size
    )
)

working_df["BID_PRICE"] = (
    working_df["NEXT_ENTRY"]
)


working_df["ENTRY_BUY"] = (
    working_df["ASK_PRICE"]
)

working_df["ENTRY_SELL"] = (
    working_df["BID_PRICE"]
)


# ===============================================================================================
# BUY TP / SL
# ===============================================================================================

working_df["SL_BUY"] = (
    working_df["ENTRY_BUY"]
    - (
        working_df["SL_PIPS"]
        * pip_size
    )
)

working_df["TP_BUY"] = (
    working_df["ENTRY_BUY"]
    + (
        working_df["TP_PIPS"]
        * pip_size
    )
)


# ===============================================================================================
# SELL TP / SL
# ===============================================================================================

working_df["SL_SELL"] = (
    working_df["ENTRY_SELL"]
    + (
        working_df["SL_PIPS"]
        * pip_size
    )
)

working_df["TP_SELL"] = (
    working_df["ENTRY_SELL"]
    - (
        working_df["TP_PIPS"]
        * pip_size
    )
)


# ===============================================================================================
# LOT SIZE
# ===============================================================================================

def calc_lot_size(
    balance,
    risk_percent,
    sl_pips,
    pip_value_per_lot,
    min_lot,
    max_lot
):

    risk_amount = (
        balance
        * (risk_percent / 100)
    )

    lot_cal = (
        risk_amount
        / (
            sl_pips
            * pip_value_per_lot
        )
    )

    lot = max(
        min_lot,
        min(
            lot_cal,
            max_lot
        )
    )

    return lot


def normalize_lot(
    lot,
    vol_min,
    vol_max,
    vol_step
):

    lot = max(
        vol_min,
        min(
            lot,
            vol_max
        )
    )

    lot = (
        np.floor(
            lot / vol_step
        )
        * vol_step
    )

    return round(
        lot,
        2
    )


# ===============================================================================================
# BACKTEST
# ===============================================================================================

balance = INITIAL_BALANCE

No_trades = 0

win_count = 0

loss_count = 0

unresolved_count = 0

total_gross_profit = 0

total_gross_loss = 0

history = []

equity_curve = []

equity_curve_dates = []


print("\n============================================================")
print("RUNNING FORWARD TEST")
print("============================================================")


i = 0

while i < len(working_df) - 1:

    row = working_df.iloc[i]

    signal = row["SIGNAL"]

    # -------------------------------------------------------
    # No signal
    # -------------------------------------------------------

    if signal == 0:

        i += 1

        continue


    # -------------------------------------------------------
    # Position sizing
    # -------------------------------------------------------

    sl_pips = row["SL_PIPS"]

    lot_size = calc_lot_size(
        balance,
        risk_percent,
        sl_pips,
        pip_value_per_lot,
        min_lot=0.01,
        max_lot=50
    )

    lot_size = normalize_lot(
        lot_size,
        vol_info["min"],
        vol_info["max"],
        vol_info["step"]
    )


    # -------------------------------------------------------
    # Trade count
    # -------------------------------------------------------

    No_trades += 1


    # -------------------------------------------------------
    # Entry
    # -------------------------------------------------------

    entry_buy = row["ENTRY_BUY"]

    entry_sell = row["ENTRY_SELL"]

    SL_buy = row["SL_BUY"]

    TP_buy = row["TP_BUY"]

    SL_sell = row["SL_SELL"]

    TP_sell = row["TP_SELL"]


    weighted_up = row["UP_AVG"]

    weighted_down = row["DN_AVG"]


    # -------------------------------------------------------
    # Signal and entry dates
    # -------------------------------------------------------

    signal_date = row["Date"]

    entry_date = row["ENTRY_DATE"]


    # -------------------------------------------------------
    # Signal hour
    # -------------------------------------------------------

    signal_hour = signal_date.hour

    entry_hour = (
        entry_date.hour
        if pd.notna(entry_date)
        else np.nan
    )


    # -------------------------------------------------------
    # Historical spread at signal/entry
    # -------------------------------------------------------

    signal_spread = row["Spread"]

    signal_spread_pips = (
        row["SPREAD_PIPS"]
    )


    # -------------------------------------------------------
    # Trade execution
    # -------------------------------------------------------

    trade_taken = False

    for j in range(
        i + 1,
        len(working_df)
    ):

        high = (
            working_df.iloc[j]["High"]
        )

        low = (
            working_df.iloc[j]["Low"]
        )


        # ===================================================
        # BUY
        # ===================================================

        if signal == 1:

            # Conservative assumption:
            # If both SL and TP occur in the
            # same candle, SL is checked first.

            if low <= SL_buy:

                loss = (
                    (
                        entry_buy
                        - SL_buy
                    )
                    / pip_size
                    * pip_value_per_lot
                    * lot_size
                )

                balance -= loss

                total_gross_loss += loss

                loss_count += 1

                equity_curve.append(
                    balance
                )

                equity_curve_dates.append(
                    entry_date
                )

                history.append({

                    "SIGNAL_DATE":
                        signal_date,

                    "ENTRY_DATE":
                        entry_date,

                    "SIGNAL_HOUR":
                        signal_hour,

                    "ENTRY_HOUR":
                        entry_hour,

                    "DIRECTION":
                        "BUY",

                    "RESULT":
                        "LOSS",

                    "SIGNAL_SPREAD":
                        signal_spread,

                    "SIGNAL_SPREAD_PIPS":
                        signal_spread_pips,

                    "UP_PROB":
                        weighted_up,

                    "DOWN_PROB":
                        weighted_down,

                    "ATR_PIPS":
                        row["ATR_PIPS"],

                    "TP_PIPS":
                        row["TP_PIPS"],

                    "SL_PIPS":
                        row["SL_PIPS"],

                    "LOT":
                        lot_size,

                    "PROFIT_LOSS":
                        -loss,

                    "BALANCE":
                        balance

                })

                i = j

                trade_taken = True

                break


            elif high >= TP_buy:

                profit = (
                    (
                        TP_buy
                        - entry_buy
                    )
                    / pip_size
                    * pip_value_per_lot
                    * lot_size
                )

                balance += profit

                total_gross_profit += profit

                win_count += 1

                equity_curve.append(
                    balance
                )

                equity_curve_dates.append(
                    entry_date
                )

                history.append({

                    "SIGNAL_DATE":
                        signal_date,

                    "ENTRY_DATE":
                        entry_date,

                    "SIGNAL_HOUR":
                        signal_hour,

                    "ENTRY_HOUR":
                        entry_hour,

                    "DIRECTION":
                        "BUY",

                    "RESULT":
                        "WIN",

                    "SIGNAL_SPREAD":
                        signal_spread,

                    "SIGNAL_SPREAD_PIPS":
                        signal_spread_pips,

                    "UP_PROB":
                        weighted_up,

                    "DOWN_PROB":
                        weighted_down,

                    "ATR_PIPS":
                        row["ATR_PIPS"],

                    "TP_PIPS":
                        row["TP_PIPS"],

                    "SL_PIPS":
                        row["SL_PIPS"],

                    "LOT":
                        lot_size,

                    "PROFIT_LOSS":
                        profit,

                    "BALANCE":
                        balance

                })

                i = j

                trade_taken = True

                break


        # ===================================================
        # SELL
        # ===================================================

        elif signal == -1:

            if high >= SL_sell:

                loss = (
                    (
                        SL_sell
                        - entry_sell
                    )
                    / pip_size
                    * pip_value_per_lot
                    * lot_size
                )

                balance -= loss

                total_gross_loss += loss

                loss_count += 1

                equity_curve.append(
                    balance
                )

                equity_curve_dates.append(
                    entry_date
                )

                history.append({

                    "SIGNAL_DATE":
                        signal_date,

                    "ENTRY_DATE":
                        entry_date,

                    "SIGNAL_HOUR":
                        signal_hour,

                    "ENTRY_HOUR":
                        entry_hour,

                    "DIRECTION":
                        "SELL",

                    "RESULT":
                        "LOSS",

                    "SIGNAL_SPREAD":
                        signal_spread,

                    "SIGNAL_SPREAD_PIPS":
                        signal_spread_pips,

                    "UP_PROB":
                        weighted_up,

                    "DOWN_PROB":
                        weighted_down,

                    "ATR_PIPS":
                        row["ATR_PIPS"],

                    "TP_PIPS":
                        row["TP_PIPS"],

                    "SL_PIPS":
                        row["SL_PIPS"],

                    "LOT":
                        lot_size,

                    "PROFIT_LOSS":
                        -loss,

                    "BALANCE":
                        balance

                })

                i = j

                trade_taken = True

                break


            elif low <= TP_sell:

                profit = (
                    (
                        entry_sell
                        - TP_sell
                    )
                    / pip_size
                    * pip_value_per_lot
                    * lot_size
                )

                balance += profit

                total_gross_profit += profit

                win_count += 1

                equity_curve.append(
                    balance
                )

                equity_curve_dates.append(
                    entry_date
                )

                history.append({

                    "SIGNAL_DATE":
                        signal_date,

                    "ENTRY_DATE":
                        entry_date,

                    "SIGNAL_HOUR":
                        signal_hour,

                    "ENTRY_HOUR":
                        entry_hour,

                    "DIRECTION":
                        "SELL",

                    "RESULT":
                        "WIN",

                    "SIGNAL_SPREAD":
                        signal_spread,

                    "SIGNAL_SPREAD_PIPS":
                        signal_spread_pips,

                    "UP_PROB":
                        weighted_up,

                    "DOWN_PROB":
                        weighted_down,

                    "ATR_PIPS":
                        row["ATR_PIPS"],

                    "TP_PIPS":
                        row["TP_PIPS"],

                    "SL_PIPS":
                        row["SL_PIPS"],

                    "LOT":
                        lot_size,

                    "PROFIT_LOSS":
                        profit,

                    "BALANCE":
                        balance

                })

                i = j

                trade_taken = True

                break


    # =======================================================
    # UNRESOLVED TRADE
    # =======================================================

    if not trade_taken:

        unresolved_count += 1

        history.append({

            "SIGNAL_DATE":
                signal_date,

            "ENTRY_DATE":
                entry_date,

            "SIGNAL_HOUR":
                signal_hour,

            "ENTRY_HOUR":
                entry_hour,

            "DIRECTION":
                "BUY" if signal == 1 else "SELL",

            "RESULT":
                "UNRESOLVED",

            "SIGNAL_SPREAD":
                signal_spread,

            "SIGNAL_SPREAD_PIPS":
                signal_spread_pips,

            "UP_PROB":
                weighted_up,

            "DOWN_PROB":
                weighted_down,

            "ATR_PIPS":
                row["ATR_PIPS"],

            "TP_PIPS":
                row["TP_PIPS"],

            "SL_PIPS":
                row["SL_PIPS"],

            "LOT":
                lot_size,

            "PROFIT_LOSS":
                0,

            "BALANCE":
                balance

        })

        i += 1


# ===============================================================================================
# HISTORY DATAFRAME
# ===============================================================================================

history_df = pd.DataFrame(
    history
)


# ===============================================================================================
# BASIC RESULTS
# ===============================================================================================

total_trades = (
    win_count
    + loss_count
)

if total_trades > 0:

    win_rate = (
        win_count
        / total_trades
        * 100
    )

else:

    win_rate = 0


if total_gross_loss > 0:

    profit_factor = (
        total_gross_profit
        / total_gross_loss
    )

else:

    profit_factor = np.inf


if total_trades > 0:

    expectancy = (
        history_df.loc[
            history_df["RESULT"] != "UNRESOLVED",
            "PROFIT_LOSS"
        ].sum()
        / total_trades
    )

else:

    expectancy = 0


# ===============================================================================================
# MAX DRAWDOWN
# ===============================================================================================

if len(equity_curve) > 0:

    equity_series = pd.Series(
        equity_curve
    )

    peak = equity_series.cummax()

    drawdown = (
        equity_series - peak
    ) / peak

    max_drawdown = (
        abs(drawdown.min())
        * 100
    )

else:

    max_drawdown = 0


# ===============================================================================================
# TRADE-LEVEL SHARPE
# ===============================================================================================

if len(equity_curve) > 1:

    returns = (
        equity_series
        .pct_change()
        .dropna()
    )

    if returns.std() != 0:

        trade_sharpe = (
            returns.mean()
            / returns.std()
        )

    else:

        trade_sharpe = np.nan

else:

    trade_sharpe = np.nan


# ===============================================================================================
# PRINT MAIN RESULTS
# ===============================================================================================

print("\n")
print("============================================================")
print("FORWARD TEST RESULTS")
print("============================================================")

print(
    f"SYMBOL             : {SYMBOL}"
)

print(
    f"START DATE         : {data['Date'].iloc[0]}"
)

print(
    f"END DATE           : {data['Date'].iloc[-1]}"
)

print(
    f"MODEL TARGET       : {HIGH_TARGETS}"
)

print(
    f"THRESHOLD          : {W_threshold}"
)

print(
    f"TP MULTIPLIER      : {tp_mult}"
)

print(
    f"SL MULTIPLIER      : {sl_mult}"
)

print(
    f"INITIAL BALANCE    : {INITIAL_BALANCE:.2f}"
)

print(
    f"FINAL BALANCE      : {balance:.2f}"
)

print(
    f"TOTAL SIGNALS      : {No_trades}"
)

print(
    f"COMPLETED TRADES   : {total_trades}"
)

print(
    f"UNRESOLVED TRADES  : {unresolved_count}"
)

print(
    f"TOTAL WINS         : {win_count}"
)

print(
    f"TOTAL LOSSES       : {loss_count}"
)

print(
    f"WIN RATE           : {win_rate:.2f}%"
)

print(
    f"PROFIT FACTOR      : {profit_factor:.2f}"
)

print(
    f"EXPECTANCY         : {expectancy:.4f}"
)

print(
    f"MAX DRAWDOWN       : {max_drawdown:.2f}%"
)

print(
    f"TRADE SHARPE       : {trade_sharpe:.2f}"
)


# ===============================================================================================
# DIRECTION ANALYSIS
# ===============================================================================================

print("\n")
print("============================================================")
print("BUY vs SELL")
print("============================================================")


for direction in ["BUY", "SELL"]:

    subset = history_df[
        history_df["DIRECTION"] == direction
    ]

    completed = subset[
        subset["RESULT"].isin(
            ["WIN", "LOSS"]
        )
    ]

    trades = len(completed)

    wins = (
        completed["RESULT"]
        == "WIN"
    ).sum()

    losses = (
        completed["RESULT"]
        == "LOSS"
    ).sum()

    if trades > 0:

        wr = (
            wins / trades * 100
        )

        gross_profit = completed.loc[
            completed["PROFIT_LOSS"] > 0,
            "PROFIT_LOSS"
        ].sum()

        gross_loss = abs(
            completed.loc[
                completed["PROFIT_LOSS"] < 0,
                "PROFIT_LOSS"
            ].sum()
        )

        pf = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf
        )

    else:

        wr = 0
        pf = np.nan

    print(
        f"\n{direction}"
    )

    print(
        f"Trades      : {trades}"
    )

    print(
        f"Wins        : {wins}"
    )

    print(
        f"Losses      : {losses}"
    )

    print(
        f"Win Rate    : {wr:.2f}%"
    )

    print(
        f"Profit Fact : {pf:.2f}"
    )


# ===============================================================================================
# HOURLY SIGNAL ANALYSIS
# ===============================================================================================

print("\n")
print("============================================================")
print("MODEL OPPORTUNITY BY SIGNAL HOUR")
print("============================================================")


hourly_rows = []


for hour in range(24):

    subset = history_df[
        history_df["SIGNAL_HOUR"] == hour
    ]

    completed = subset[
        subset["RESULT"].isin(
            ["WIN", "LOSS"]
        )
    ]

    signals = len(subset)

    wins = (
        completed["RESULT"]
        == "WIN"
    ).sum()

    losses = (
        completed["RESULT"]
        == "LOSS"
    ).sum()

    completed_trades = (
        wins + losses
    )

    if completed_trades > 0:

        hourly_win_rate = (
            wins
            / completed_trades
            * 100
        )

        gross_profit = completed.loc[
            completed["PROFIT_LOSS"] > 0,
            "PROFIT_LOSS"
        ].sum()

        gross_loss = abs(
            completed.loc[
                completed["PROFIT_LOSS"] < 0,
                "PROFIT_LOSS"
            ].sum()
        )

        pf = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf
        )

        hourly_expectancy = (
            completed["PROFIT_LOSS"].mean()
        )

    else:

        hourly_win_rate = 0
        pf = np.nan
        hourly_expectancy = 0


    buy_count = (
        subset["DIRECTION"]
        == "BUY"
    ).sum()

    sell_count = (
        subset["DIRECTION"]
        == "SELL"
    ).sum()


    avg_spread = (
        subset["SIGNAL_SPREAD_PIPS"]
        .mean()
        if len(subset) > 0
        else np.nan
    )

    median_spread = (
        subset["SIGNAL_SPREAD_PIPS"]
        .median()
        if len(subset) > 0
        else np.nan
    )

    max_spread = (
        subset["SIGNAL_SPREAD_PIPS"]
        .max()
        if len(subset) > 0
        else np.nan
    )


    hourly_rows.append({

        "HOUR":
            hour,

        "SIGNALS":
            signals,

        "BUY":
            buy_count,

        "SELL":
            sell_count,

        "WINS":
            wins,

        "LOSSES":
            losses,

        "WIN_RATE":
            hourly_win_rate,

        "PROFIT_FACTOR":
            pf,

        "EXPECTANCY":
            hourly_expectancy,

        "AVG_SPREAD_PIPS":
            avg_spread,

        "MEDIAN_SPREAD_PIPS":
            median_spread,

        "MAX_SPREAD_PIPS":
            max_spread

    })


hourly_df = pd.DataFrame(
    hourly_rows
)


# ===============================================================================================
# PRINT HOURLY TABLE
# ===============================================================================================

print(
    hourly_df.to_string(
        index=False,
        formatters={
            "WIN_RATE":
                lambda x: f"{x:.2f}%",
            "PROFIT_FACTOR":
                lambda x:
                f"{x:.2f}"
                if pd.notna(x)
                else "-",
            "EXPECTANCY":
                lambda x: f"{x:.4f}",
            "AVG_SPREAD_PIPS":
                lambda x:
                f"{x:.2f}"
                if pd.notna(x)
                else "-",
            "MEDIAN_SPREAD_PIPS":
                lambda x:
                f"{x:.2f}"
                if pd.notna(x)
                else "-",
            "MAX_SPREAD_PIPS":
                lambda x:
                f"{x:.2f}"
                if pd.notna(x)
                else "-"
        }
    )
)


# ===============================================================================================
# PROBLEM PERIOD: 8PM - 10PM
# ===============================================================================================

print("\n")
print("============================================================")
print("8 PM - 10 PM ANALYSIS")
print("============================================================")


# 20:00, 21:00, 22:00
problem_hours = [20, 21, 22]

problem_period = history_df[
    history_df["SIGNAL_HOUR"].isin(
        problem_hours
    )
]

problem_completed = problem_period[
    problem_period["RESULT"].isin(
        ["WIN", "LOSS"]
    )
]

problem_trades = len(
    problem_completed
)

problem_wins = (
    problem_completed["RESULT"]
    == "WIN"
).sum()

problem_losses = (
    problem_completed["RESULT"]
    == "LOSS"
).sum()


if problem_trades > 0:

    problem_win_rate = (
        problem_wins
        / problem_trades
        * 100
    )

    problem_profit = (
        problem_completed.loc[
            problem_completed["PROFIT_LOSS"] > 0,
            "PROFIT_LOSS"
        ].sum()
    )

    problem_loss = abs(
        problem_completed.loc[
            problem_completed["PROFIT_LOSS"] < 0,
            "PROFIT_LOSS"
        ].sum()
    )

    problem_pf = (
        problem_profit / problem_loss
        if problem_loss > 0
        else np.inf
    )

else:

    problem_win_rate = 0
    problem_pf = np.nan


print(
    f"Signals          : {len(problem_period)}"
)

print(
    f"Completed Trades : {problem_trades}"
)

print(
    f"Wins             : {problem_wins}"
)

print(
    f"Losses           : {problem_losses}"
)

print(
    f"Win Rate         : {problem_win_rate:.2f}%"
)

print(
    f"Profit Factor    : {problem_pf:.2f}"
)

print(
    f"Average Spread   : "
    f"{problem_period['SIGNAL_SPREAD_PIPS'].mean():.2f} pips"
)

print(
    f"Median Spread    : "
    f"{problem_period['SIGNAL_SPREAD_PIPS'].median():.2f} pips"
)

print(
    f"Maximum Spread   : "
    f"{problem_period['SIGNAL_SPREAD_PIPS'].max():.2f} pips"
)


# ===============================================================================================
# 8PM-10PM EACH HOUR
# ===============================================================================================

print("\n")
print("------------------------------------------------------------")
print("8 PM / 9 PM / 10 PM BREAKDOWN")
print("------------------------------------------------------------")


print(
    hourly_df[
        hourly_df["HOUR"].isin(
            problem_hours
        )
    ].to_string(
        index=False
    )
)


# ===============================================================================================
# OTHER PERIODS
# ===============================================================================================

periods = {

    "00:00-05:59": list(range(0, 6)),

    "06:00-11:59": list(range(6, 12)),

    "12:00-17:59": list(range(12, 18)),

    "18:00-19:59": [18, 19],

    "20:00-22:00": [20, 21, 22],

    "23:00-23:59": [23]

}


period_rows = []


print("\n")
print("============================================================")
print("TIME PERIOD COMPARISON")
print("============================================================")


for period_name, hours in periods.items():

    subset = history_df[
        history_df["SIGNAL_HOUR"].isin(hours)
    ]

    completed = subset[
        subset["RESULT"].isin(
            ["WIN", "LOSS"]
        )
    ]

    trades = len(completed)

    wins = (
        completed["RESULT"]
        == "WIN"
    ).sum()

    losses = (
        completed["RESULT"]
        == "LOSS"
    ).sum()

    if trades > 0:

        wr = (
            wins / trades * 100
        )

        gp = completed.loc[
            completed["PROFIT_LOSS"] > 0,
            "PROFIT_LOSS"
        ].sum()

        gl = abs(
            completed.loc[
                completed["PROFIT_LOSS"] < 0,
                "PROFIT_LOSS"
            ].sum()
        )

        pf = (
            gp / gl
            if gl > 0
            else np.inf
        )

        expectancy_period = (
            completed["PROFIT_LOSS"].mean()
        )

    else:

        wr = 0
        pf = np.nan
        expectancy_period = 0


    period_rows.append({

        "PERIOD":
            period_name,

        "SIGNALS":
            len(subset),

        "TRADES":
            trades,

        "WINS":
            wins,

        "LOSSES":
            losses,

        "WIN_RATE":
            wr,

        "PROFIT_FACTOR":
            pf,

        "EXPECTANCY":
            expectancy_period,

        "AVG_SPREAD_PIPS":
            subset["SIGNAL_SPREAD_PIPS"].mean()
            if len(subset) > 0
            else np.nan

    })


period_df = pd.DataFrame(
    period_rows
)


print(
    period_df.to_string(
        index=False
    )
)


# ===============================================================================================
# MONTHLY ANALYSIS
# ===============================================================================================

history_df["MONTH"] = (
    history_df["SIGNAL_DATE"]
    .dt.to_period("M")
    .astype(str)
)


monthly_rows = []


for month, subset in history_df.groupby(
    "MONTH"
):

    completed = subset[
        subset["RESULT"].isin(
            ["WIN", "LOSS"]
        )
    ]

    trades = len(completed)

    wins = (
        completed["RESULT"]
        == "WIN"
    ).sum()

    losses = (
        completed["RESULT"]
        == "LOSS"
    ).sum()

    if trades > 0:

        wr = (
            wins / trades * 100
        )

        gp = completed.loc[
            completed["PROFIT_LOSS"] > 0,
            "PROFIT_LOSS"
        ].sum()

        gl = abs(
            completed.loc[
                completed["PROFIT_LOSS"] < 0,
                "PROFIT_LOSS"
            ].sum()
        )

        pf = (
            gp / gl
            if gl > 0
            else np.inf
        )

        pnl = (
            completed["PROFIT_LOSS"]
            .sum()
        )

    else:

        wr = 0
        pf = np.nan
        pnl = 0


    monthly_rows.append({

        "MONTH":
            month,

        "TRADES":
            trades,

        "WINS":
            wins,

        "LOSSES":
            losses,

        "WIN_RATE":
            wr,

        "PROFIT_FACTOR":
            pf,

        "PNL":
            pnl

    })


monthly_df = pd.DataFrame(
    monthly_rows
)


print("\n")
print("============================================================")
print("MONTHLY RESULTS")
print("============================================================")


print(
    monthly_df.to_string(
        index=False
    )
)


# ===============================================================================================
# SIGNAL DISTRIBUTION
# ===============================================================================================

print("\n")
print("============================================================")
print("SIGNAL DISTRIBUTION")
print("============================================================")


signal_hour_counts = (
    history_df[
        history_df["RESULT"] != "UNRESOLVED"
    ]
    ["SIGNAL_HOUR"]
    .value_counts()
    .sort_index()
)


print(
    signal_hour_counts.to_string()
)


# ===============================================================================================
# SAVE RESULTS
# ===============================================================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "EURUSD_forward_trade_history.csv"
)

hourly_path = os.path.join(
    OUTPUT_DIR,
    "EURUSD_opportunity_by_hour.csv"
)

period_path = os.path.join(
    OUTPUT_DIR,
    "EURUSD_opportunity_by_period.csv"
)

monthly_path = os.path.join(
    OUTPUT_DIR,
    "EURUSD_monthly_results.csv"
)


history_df.to_csv(
    history_path,
    index=False
)

hourly_df.to_csv(
    hourly_path,
    index=False
)

period_df.to_csv(
    period_path,
    index=False
)

monthly_df.to_csv(
    monthly_path,
    index=False
)


# ===============================================================================================
# FINAL MESSAGE
# ===============================================================================================

print("\n")
print("============================================================")
print("FILES SAVED")
print("============================================================")

print(
    f"Trade history :\n{history_path}"
)

print(
    f"\nHourly analysis :\n{hourly_path}"
)

print(
    f"\nPeriod analysis :\n{period_path}"
)

print(
    f"\nMonthly results :\n{monthly_path}"
)


print("\n")
print("============================================================")
print("ANALYSIS COMPLETE")
print("============================================================")