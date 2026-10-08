
# ===============================================================================================
# BINARY MODEL — REALISTIC FORWARD TEST / THRESHOLD FINDER
# ===============================================================================================


import json
import os
import joblib
import numpy as np
import pandas as pd

from func import apply_features, normalize_symbol


# ===============================================================================================
# CONFIGURATION
# ===============================================================================================

SYMBOL = "NZDUSD"
TRADE_NO = 225
CD_TIME = "10M"

risk_percent = 1

tp_mult = 1.75
sl_mult = 1.75

# LIVE EXECUTION RULE
SPREAD_PERCENT = 0.15

# Same minimum stop-distance concept as live execution.
# Your live filter currently uses:
#     symbol_info.point * 10
MIN_STOP_DISTANCE_POINTS = 10


HIGH_TARGETS = ["THL_3H"]

weights = {"THL_3H": 1.0}


# ===============================================================================================
# PATHS
# ===============================================================================================

BASE_PATH = "/content/drive/MyDrive/FOREX TRADING/ALL_MODELS"

MODEL_FOLDER = "/content/drive/MyDrive/FOREX TRADING/ALL_MODELS"
CSV_FOLDER = "/content/drive/MyDrive/FOREX TRADING/CSV_FILES"
IMPORTANCE_FOLDER = "/content/drive/MyDrive/FOREX TRADING/SHAP RESULT"


# ===============================================================================================
# LOAD MODELS
# ===============================================================================================

def load_models(model_type, CD_TIME, symbol):

    symbol = normalize_symbol(symbol=symbol)

    models_dict = {}

    for target in HIGH_TARGETS:

        file_path = (f"{BASE_PATH}/HL_{model_type}_{target}_{CD_TIME}_{symbol}_model.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model not found: {file_path}")

        bundle = joblib.load(file_path)

        models_dict[target] = {
            "model": bundle["model"],
            "features": bundle["features"]
        }

        print(f"\n [[GOOD]] Loaded {model_type} model for {target}")

    return models_dict


# ===============================================================================================
# LOAD MODEL
# ===============================================================================================

HELPER_MODELS = load_models("MAIN",CD_TIME,SYMBOL)


# ===============================================================================================
# LOAD SYMBOL INFORMATION
# ===============================================================================================

with open("/content/drive/MyDrive/FOREX TRADING/CSV_FILES/SYMBOL_INFO.json") as f:

    SYMBOL_INFO = json.load(f)


info = SYMBOL_INFO[SYMBOL]

balance = info["balance"]
equity = info["equity"]
currency = info["currency"]
leverage = info["leverage"]

pip_size = info["pip_size"]
pip_value_per_lot = info["pip_value_per_lot"]

min_gap = info["min_gap"]

vol_info = info["vol_info"]

min_lot = vol_info["min"]
max_lot = vol_info["max"]
lot_step = vol_info["step"]


# ===============================================================================================
# LOAD POINT INFORMATION
# ===============================================================================================

POINTS_FILE = os.path.join(CSV_FOLDER,"symbol_points.json")

MT5_SYMBOL = SYMBOL + "c"

print(f"\nCURRENTLY RUNNING SYMBOL: {SYMBOL}")
print(f"MT5 SYMBOL: {MT5_SYMBOL}")


with open( POINTS_FILE, "r", encoding="utf-8") as file:

    SYMBOL_POINTS = json.load(file)


if MT5_SYMBOL not in SYMBOL_POINTS:

    raise ValueError(f"Point information not found for {MT5_SYMBOL} in {POINTS_FILE}")


POINT = SYMBOL_POINTS[MT5_SYMBOL]["point"]

print(f"POINT: {POINT}")


# ===============================================================================================
# LOAD DATA
# ===============================================================================================

DATA_FILE = (f"{CSV_FOLDER}/MT5_{CD_TIME}_BT_{SYMBOL}_Exchange_Rate_Dataset.csv")

data = pd.read_csv(DATA_FILE)

print("\n\t DATASET LOADED SUCCESSFULLY")
print("\t STARTING APPLY FEATURES")

# ===============================================================================================
# APPLY FEATURES
# ===============================================================================================

df = apply_features( df=data, point=POINT)


# ===============================================================================================
# FORWARD TEST PERIOD
# ===============================================================================================

# df = df[ (df["Date"] >= "2026-01-01") & (df["Date"] < "2026-10-01")].copy()

# print("FORWARD TEST START:", df["Date"].min())
# print("FORWARD TEST END:", df["Date"].max())


# ===============================================================================================
# CLEAN DATA
# ===============================================================================================

df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

print("\nDF INFO:",df.shape)
print("\t APPLY FEATURE SUCCESSFULLY")


# ===============================================================================================
# WORKING DATAFRAME
# ===============================================================================================
working_df = df[["Date","High","Low","Open","Close","ATR","Spread"]].copy()


# ===============================================================================================
# MODEL PREDICTIONS
# ===============================================================================================

for target in HIGH_TARGETS:

    print(f"\n [[ CURRENTLY PREDICTING TARGET : {target} ]]")

    help_model = HELPER_MODELS[target]["model"]

    help_cols = HELPER_MODELS[target]["features"]

    X_help = df[help_cols]

    # ------------------------------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------------------------------
    assert set(help_cols) == set(X_help.columns)

    # ------------------------------------------------------------------
    # MODEL PROBABILITIES
    # ------------------------------------------------------------------
    help_proba = help_model.predict_proba(X_help)

    up_prob = help_proba[:, 1]

    down_prob = help_proba[:, 0]

    working_df[f"{target}_UP"] = up_prob
    working_df[f"{target}_DN"] = down_prob


# ===============================================================================================
# WEIGHTED MODEL PROBABILITIES
# ===============================================================================================

weight_list = [ weights[t] for t in HIGH_TARGETS]
total_weight = sum( weight_list)

working_df["UP_AVG"] = ( sum( working_df[f"{t}_UP"] * weights[t] for t in HIGH_TARGETS ) / total_weight)
working_df["DN_AVG"] = ( sum( working_df[f"{t}_DN"] * weights[t] for t in HIGH_TARGETS) / total_weight)

# ===============================================================================================
# ATR / SL / TP
# ===============================================================================================

working_df["ATR_PIPS"] = ( working_df["ATR"] / pip_size)
working_df["SL_PIPS"] = ( working_df["ATR_PIPS"] * sl_mult)
working_df["TP_PIPS"] = ( working_df["ATR_PIPS"] * tp_mult)

# Spread_Price = Spread_Points * POINT

working_df["SPREAD_PRICE"] = ( working_df["Spread"] * POINT)

working_df["NEXT_ENTRY"] = ( working_df["Open"].shift(-1))
working_df["ASK_PRICE"] = ( working_df["NEXT_ENTRY"] + working_df["SPREAD_PRICE"])
working_df["BID_PRICE"] = ( working_df["NEXT_ENTRY"])
working_df["ENTRY_BUY"] = ( working_df["ASK_PRICE"])
working_df["ENTRY_SELL"] = ( working_df["BID_PRICE"])


# ===============================================================================================
# SL / TP
# ===============================================================================================

# BUY
working_df["SL_BUY"] = ( working_df["ENTRY_BUY"] - ( working_df["SL_PIPS"] * pip_size ))
working_df["TP_BUY"] = ( working_df["ENTRY_BUY"] + ( working_df["TP_PIPS"] *pip_size ))

# SELL
working_df["SL_SELL"] = ( working_df["ENTRY_SELL"] + ( working_df["SL_PIPS"] * pip_size ))
working_df["TP_SELL"] = ( working_df["ENTRY_SELL"] - ( working_df["TP_PIPS"] * pip_size))

# ===============================================================================================
# The live rule is:
#     spread / SL_distance <= 15%
#     SL_distance = 1.75 * ATR
#
# We calculate:
#     SPREAD_RATIO = actual spread price / actual SL distance
# ===============================================================================================

working_df["SL_DISTANCE"] = ( sl_mult * working_df["ATR"])
working_df["SPREAD_RATIO"] = ( working_df["SPREAD_PRICE"] / ( working_df["SL_DISTANCE"] + 1e-9 ))

working_df["SPREAD_EXECUTION_OK"] = ( working_df["SPREAD_RATIO"] <= SPREAD_PERCENT)

# ===============================================================================================
# MINIMUM STOP DISTANCE CHECK
# Same basic concept as your live filter:
#     SL_DISTANCE >= POINT * 10
# ===============================================================================================

MIN_STOP_DISTANCE = (POINT * MIN_STOP_DISTANCE_POINTS)

working_df["STOP_DISTANCE_OK"] = ( working_df["SL_DISTANCE"] >= MIN_STOP_DISTANCE)


# ===============================================================================================
# FINAL EXECUTION CONDITION
# ===============================================================================================

working_df["EXECUTION_OK"] = ( working_df["SPREAD_EXECUTION_OK"] & working_df["STOP_DISTANCE_OK"])


# ===============================================================================================
# LOT SIZE
# ===============================================================================================

def calc_lot_size(balance,risk_percent,sl_pips,pip_value_per_lot,min_lot,max_lot):

    risk_amount = (balance *(risk_percent / 100))
    lot_cal = (risk_amount /(sl_pips * pip_value_per_lot))
    lot = max( min_lot, min( lot_cal, max_lot ))

    return lot


# ===============================================================================================
# NORMALIZE LOT
# ===============================================================================================

def normalize_lot(lot,vol_min,vol_max,vol_step):

    lot = max( vol_min, min( lot, vol_max))
    lot = ( np.floor( lot / vol_step ) * vol_step)

    return round( lot, 2)

# ===============================================================================================
# BACKTEST ONE THRESHOLD
# ===============================================================================================

def run_backtest(working_df, threshold):

    # ===========================================================================================
    # NUMPY ARRAYS
    # ===========================================================================================

    up = working_df["UP_AVG"].to_numpy()
    dn = working_df["DN_AVG"].to_numpy()

    # ------------------------------------------------------------------
    # TIME INFORMATION
    # ------------------------------------------------------------------

    dates = pd.to_datetime(working_df["Date"]).to_numpy()
    execution_ok = (working_df["EXECUTION_OK"].to_numpy())

    high = (working_df["High"].to_numpy())
    low = (working_df["Low"].to_numpy())
    sl_buy = (working_df["SL_BUY"].to_numpy())
    tp_buy = (working_df["TP_BUY"].to_numpy())
    entry_buy = (working_df["ENTRY_BUY"].to_numpy())
    sl_sell = ( working_df["SL_SELL"].to_numpy())
    tp_sell = ( working_df["TP_SELL"].to_numpy())
    entry_sell = ( working_df["ENTRY_SELL"].to_numpy())
    sl_pips_array = ( working_df["SL_PIPS"].to_numpy())
    spread_ratio = ( working_df["SPREAD_RATIO"].to_numpy())


    # ===========================================================================================
    # MODEL SIGNALS
    # ===========================================================================================

    signals = np.zeros( len(working_df), dtype=np.int8)

    buy_condition = ( (up >= threshold) & (up > dn))
    sell_condition = ( (dn >= threshold) & (dn > up))

    signals[buy_condition] = 1
    signals[sell_condition] = -1


    # ===========================================================================================
    # MODEL SIGNAL COUNT
    # ===========================================================================================
    model_signal_mask = ( signals != 0)
    model_signals = int(model_signal_mask.sum())

    # ===========================================================================================
    # EXECUTION AVAILABILITY
    # ===========================================================================================

    executable_mask = ( model_signal_mask & execution_ok)
    executable_trades_available = int( executable_mask.sum())

    # -------------------------------------------------------------------------------------------
    # SIGNALS THAT FAIL THE EXECUTION FILTER
    # -------------------------------------------------------------------------------------------
    spread_rejected = int((model_signal_mask & ~execution_ok).sum())


    # ===========================================================================================
    # SIGNAL LOG
    #
    # This is kept completely separate from the threshold result.
    #
    # Every model signal will eventually receive exactly ONE status.
    # ===========================================================================================

    signal_status = np.full(len(working_df), "", dtype=object)

    signal_entry_time = np.full(len(working_df),np.datetime64("NaT"),dtype="datetime64[ns]")
    signal_exit_time = np.full(len(working_df),np.datetime64("NaT"),dtype="datetime64[ns]")

    signal_trade_result = np.full(len(working_df),"",dtype=object)


    # ===========================================================================================
    # ACCOUNT VARIABLES
    # ===========================================================================================

    threshold_balance = balance

    starting_balance = balance

    wins = 0

    losses = 0

    total_gross_profit = 0.0

    total_gross_loss = 0.0

    equity_curve = []

    trade_returns = []


    # ===========================================================================================
    # SIGNAL ACCOUNTING
    # ===========================================================================================

    overlapping_signals = 0

    unresolved_signals = 0


    # ===========================================================================================
    # BACKTEST LOOP
    # ===========================================================================================

    i = 0

    n = len(signals)


    while i < n - 1:

        signal = signals[i]


        # ---------------------------------------------------------------------------------------
        # NO MODEL SIGNAL
        # ---------------------------------------------------------------------------------------

        if signal == 0:

            i += 1

            continue


        # ---------------------------------------------------------------------------------------
        # MODEL SIGNAL EXISTS BUT TRADE IS NOT EXECUTABLE
        # ---------------------------------------------------------------------------------------

        if not execution_ok[i]:

            signal_status[i] = "SPREAD_REJECTED"

            i += 1

            continue


        # =======================================================================================
        # LOT SIZE
        # =======================================================================================

        current_sl_pips = sl_pips_array[i]


        lot_size = calc_lot_size(threshold_balance,risk_percent,current_sl_pips,
            pip_value_per_lot,min_lot=min_lot,max_lot=max_lot)

        lot_size = normalize_lot(lot_size,vol_info["min"],vol_info["max"],vol_info["step"])

        balance_before_trade = threshold_balance


        # =======================================================================================
        # BUY
        # =======================================================================================

        if signal == 1:

            sl = sl_buy[i]

            tp = tp_buy[i]

            entry = entry_buy[i]


            j = i + 1


            while j < n:

                # --------------------------------------------------------------------------------
                # STOP LOSS
                # --------------------------------------------------------------------------------

                if low[j] <= sl:

                    loss_amount = ( ( (entry - sl ) / pip_size ) * pip_value_per_lot * lot_size)

                    threshold_balance -= loss_amount

                    total_gross_loss += loss_amount

                    losses += 1


                    trade_return = (( threshold_balance -  balance_before_trade ) / balance_before_trade)

                    trade_returns.append(trade_return)
                    equity_curve.append(threshold_balance)

                    # --------------------------------------------------------------------------
                    # CURRENT SIGNAL
                    # --------------------------------------------------------------------------

                    signal_status[i] = "EXECUTED"

                    signal_entry_time[i] = dates[i]

                    signal_exit_time[i] = dates[j]

                    signal_trade_result[i] = "LOSS"


                    # --------------------------------------------------------------------------
                    # EXECUTABLE SIGNALS THAT OCCURRED WHILE TRADE WAS OPEN
                    # --------------------------------------------------------------------------
                    overlapping_indices = np.where( executable_mask[ i + 1 : j])[0] + i + 1

                    for k in overlapping_indices:

                        signal_status[k] = "OVERLAPPING"

                    overlapping_signals += len(overlapping_indices)


                    i = j

                    break


                # --------------------------------------------------------------------------------
                # TAKE PROFIT
                # --------------------------------------------------------------------------------

                if high[j] >= tp:

                    profit_amount = ( ( (  tp - entry ) / pip_size ) * pip_value_per_lot * lot_size)

                    threshold_balance += profit_amount

                    total_gross_profit += profit_amount

                    wins += 1


                    trade_return = ( ( threshold_balance - balance_before_trade ) / balance_before_trade)

                    trade_returns.append( trade_return)
                    equity_curve.append( threshold_balance)

                    # --------------------------------------------------------------------------
                    # CURRENT SIGNAL
                    # --------------------------------------------------------------------------

                    signal_status[i] = "EXECUTED"

                    signal_entry_time[i] = dates[i]

                    signal_exit_time[i] = dates[j]

                    signal_trade_result[i] = "WIN"


                    # --------------------------------------------------------------------------
                    # EXECUTABLE SIGNALS THAT OCCURRED WHILE TRADE WAS OPEN
                    # --------------------------------------------------------------------------
                    overlapping_indices = np.where(executable_mask[ i + 1 : j ])[0] + i + 1

                    for k in overlapping_indices:

                        signal_status[k] = "OVERLAPPING"

                    overlapping_signals += len(overlapping_indices)

                    i = j

                    break


                j += 1


            # -----------------------------------------------------------------------------------
            # TRADE REACHED END OF DATA WITHOUT TP OR SL
            # -----------------------------------------------------------------------------------

            if j == n:

                signal_status[i] = "UNRESOLVED"

                unresolved_signals += 1

                # -------------------------------------------------------------------------------
                # Any later executable signals also cannot become completed trades.
                # -------------------------------------------------------------------------------

                unresolved_indices = np.where( executable_mask[ i + 1 : n ] )[0] + i + 1

                for k in unresolved_indices:

                    signal_status[k] = "UNRESOLVED"

                unresolved_signals += len( unresolved_indices )

                break


        # =======================================================================================
        # SELL
        # =======================================================================================

        elif signal == -1:

            sl = sl_sell[i]

            tp = tp_sell[i]

            entry = entry_sell[i]


            j = i + 1


            while j < n:

                # --------------------------------------------------------------------------------
                # STOP LOSS
                # --------------------------------------------------------------------------------

                if high[j] >= sl:

                    loss_amount = ( (( sl - entry) / pip_size ) * pip_value_per_lot * lot_size)

                    threshold_balance -= loss_amount

                    total_gross_loss += loss_amount

                    losses += 1

                    trade_return = ( ( threshold_balance - balance_before_trade ) / balance_before_trade)

                    trade_returns.append(trade_return)
                    equity_curve.append(threshold_balance)

                    # --------------------------------------------------------------------------
                    # CURRENT SIGNAL
                    # --------------------------------------------------------------------------

                    signal_status[i] = "EXECUTED"

                    signal_entry_time[i] = dates[i]

                    signal_exit_time[i] = dates[j]

                    signal_trade_result[i] = "LOSS"


                    # --------------------------------------------------------------------------
                    # EXECUTABLE SIGNALS THAT OCCURRED WHILE TRADE WAS OPEN
                    # --------------------------------------------------------------------------

                    overlapping_indices = np.where(
                        executable_mask[ i + 1 : j] )[0] + i + 1

                    for k in overlapping_indices:

                        signal_status[k] = "OVERLAPPING"

                    overlapping_signals += len( overlapping_indices)

                    i = j

                    break


                # --------------------------------------------------------------------------------
                # TAKE PROFIT
                # --------------------------------------------------------------------------------

                if low[j] <= tp:

                    profit_amount = ( ( ( entry -  tp  ) / pip_size) * pip_value_per_lot * lot_size)

                    threshold_balance += profit_amount

                    total_gross_profit += profit_amount

                    wins += 1

                    trade_return = ( (  threshold_balance - balance_before_trade ) / balance_before_trade )

                    trade_returns.append(trade_return)
                    equity_curve.append( threshold_balance)


                    # --------------------------------------------------------------------------
                    # CURRENT SIGNAL
                    # --------------------------------------------------------------------------

                    signal_status[i] = "EXECUTED"

                    signal_entry_time[i] = dates[i]

                    signal_exit_time[i] = dates[j]

                    signal_trade_result[i] = "WIN"


                    # --------------------------------------------------------------------------
                    # EXECUTABLE SIGNALS THAT OCCURRED WHILE TRADE WAS OPEN
                    # --------------------------------------------------------------------------

                    overlapping_indices = np.where( executable_mask[ i + 1 : j ] )[0] + i + 1

                    for k in overlapping_indices:
                        signal_status[k] = "OVERLAPPING"

                    overlapping_signals += len( overlapping_indices)

                    i = j

                    break


                j += 1


            # -----------------------------------------------------------------------------------
            # TRADE REACHED END OF DATA WITHOUT TP OR SL
            # -----------------------------------------------------------------------------------

            if j == n:

                signal_status[i] = "UNRESOLVED"

                unresolved_signals += 1

                unresolved_indices = np.where(
                    executable_mask[ i + 1 : n ])[0] + i + 1

                for k in unresolved_indices:

                    signal_status[k] = "UNRESOLVED"

                unresolved_signals += len(unresolved_indices)

                break


        # ---------------------------------------------------------------------------------------
        # SAFETY
        # ---------------------------------------------------------------------------------------

        else:

            i += 1


    # ===========================================================================================
    # SAFETY: CLASSIFY ANY MODEL SIGNAL THAT HAS NOT YET RECEIVED A STATUS
    # ===========================================================================================

    remaining_signal_indices = np.where(
        model_signal_mask &
        (signal_status == "")
    )[0]


    for k in remaining_signal_indices:

        if not execution_ok[k]:

            signal_status[k] = "SPREAD_REJECTED"

        else:

            signal_status[k] = "OVERLAPPING"


            # This is only a safety fallback.
            # Normally every executable signal should already have been
            # classified during the chronological backtest.


    # ===========================================================================================
    # CREATE SEPARATE SIGNAL LOG DATAFRAME
    # ===========================================================================================

    signal_indices = np.where(model_signal_mask)[0]

    signal_log_df = pd.DataFrame({

        "THRESHOLD": threshold,
        "SIGNAL_TIME": dates[signal_indices],
        "SIGNAL": np.where( signals[signal_indices] == 1, "BUY", "SELL" ),
        "UP_AVG": up[signal_indices],
        "DN_AVG": dn[signal_indices],
        "EXECUTION_OK": execution_ok[signal_indices],
        "SPREAD_RATIO": spread_ratio[signal_indices],
        "STATUS": signal_status[signal_indices],
        "ENTRY_TIME": signal_entry_time[signal_indices],
        "EXIT_TIME": signal_exit_time[signal_indices],
        "TRADE_RESULT": signal_trade_result[signal_indices]
    })


    # ===========================================================================================
    # TOTAL EXECUTED TRADES
    # ===========================================================================================
    total_trades = ( wins + losses)


    # ===========================================================================================
    # WIN RATE
    # ===========================================================================================

    if total_trades > 0:
        win_rate = ( wins / total_trades)

    else:
        win_rate = 0.0


    # ===========================================================================================
    # PROFIT FACTOR
    # ===========================================================================================

    if total_gross_loss > 0:
        profit_factor = ( total_gross_profit / total_gross_loss)

    else:
        profit_factor = float("inf")


    # ===========================================================================================
    # EXPECTANCY
    # ===========================================================================================

    if wins > 0:
        avg_win = ( total_gross_profit / wins )

    else:
        avg_win = 0.0


    if losses > 0:
        avg_loss = ( total_gross_loss / losses)

    else:
        avg_loss = 0.0

    win_probability = ( wins / total_trades if total_trades > 0 else 0)
    loss_probability = ( losses / total_trades if total_trades > 0 else 0)

    expectancy = ( ( win_probability * avg_win ) - ( loss_probability * avg_loss ))

    # ===========================================================================================
    # MAXIMUM DRAWDOWN
    # ===========================================================================================

    if len(equity_curve) > 0:

        equity_series = pd.Series(equity_curve)
        running_peak = ( equity_series.cummax())
        drawdowns = ( ( running_peak - equity_series ) / running_peak)

        max_drawdown = ( drawdowns.max())

    else:
        max_drawdown = 0.0


    # ===========================================================================================
    # TRADE SHARPE
    # ===========================================================================================

    if len(trade_returns) > 1:
        trade_returns_array = np.asarray( trade_returns,dtype=np.float64)

        returns_std = ( trade_returns_array.std(ddof=1))

        if returns_std > 0:
            trade_sharpe = ( trade_returns_array.mean() / returns_std ) * np.sqrt(252)

        else:
            trade_sharpe = 0.0

    else:
        trade_sharpe = 0.0


    # ===========================================================================================
    # ACCOUNTING CHECK
    # ===========================================================================================

    accounted_signals = ( total_trades + spread_rejected + overlapping_signals + unresolved_signals)

    accounting_difference = (model_signals - accounted_signals)

    # ===========================================================================================
    # FINAL RESULT
    # ===========================================================================================

    result = {

        # ------------------------------------------------------------------
        # THRESHOLD
        # ------------------------------------------------------------------
        "THRESHOLD": round( threshold, 3 ),

        # ------------------------------------------------------------------
        # MODEL WORLD
        # ------------------------------------------------------------------
        "MODEL_SIGNALS":model_signals,

        # ------------------------------------------------------------------
        # EXECUTION WORLD
        # ------------------------------------------------------------------

        "EXECUTABLE_TRADES_AVAILABLE":executable_trades_available,
        "EXECUTABLE_TRADES":total_trades,
        "SPREAD_REJECTED": spread_rejected,
        "OVERLAPPING_SIGNALS": overlapping_signals,
        "UNRESOLVED_SIGNALS": unresolved_signals,
        "ACCOUNTED_SIGNALS": accounted_signals,
        "ACCOUNTING_DIFFERENCE": accounting_difference,


        # ------------------------------------------------------------------
        # PERFORMANCE
        # ------------------------------------------------------------------
        "WINS": wins,
        "LOSSES": losses,
        "WIN_RATE": round( win_rate * 100, 2 ),
        "PROFIT_FACTOR": (
                round(
                    profit_factor,
                    2
                )
                if np.isfinite(
                    profit_factor
                )
                else np.inf
            ),

        "EXPECTANCY": round(expectancy, 2),
        "MAX_DRAWDOWN": round( max_drawdown * 100, 2 ),
        "TRADE_SHARPE": round( trade_sharpe, 2),


        # ------------------------------------------------------------------
        # BALANCE
        # ------------------------------------------------------------------
        "STARTING_BALANCE":round(starting_balance, 2 ),
        "FINAL_BALANCE": round(threshold_balance, 2 ),
        "GROSS_PROFIT": round( total_gross_profit,2),
        "GROSS_LOSS":round( total_gross_loss, 2)
    }


    # ===========================================================================================
    # RETURN BOTH DATASETS SEPARATELY
    # ===========================================================================================

    return result, signal_log_df


# ===============================================================================================
# THRESHOLD GENERATION
# ===============================================================================================

results = []

signal_logs = []

thresholds = sorted(set(pd.concat([working_df["UP_AVG"], working_df["DN_AVG"] ]).round(3).dropna()))

print("\nTHRESHOLD LENGTH >>",len(thresholds))


# ===============================================================================================
# RUN ALL THRESHOLDS
# ===============================================================================================

for counter, threshold in enumerate(thresholds,start=1):

    print(f"\nCURRENTLY ON {threshold} THRESHOLD >> {counter}/{len(thresholds)}")

    result, signal_log = run_backtest(working_df,threshold)
    results.append(result)

    signal_logs.append(signal_log)


# ===============================================================================================
# RESULT DATAFRAME
# ===============================================================================================
result_df = pd.DataFrame(results)

# ===============================================================================================
# COMBINE SIGNAL LOGS INTO A SEPARATE DATAFRAME
# ===============================================================================================

signal_log_df = pd.concat(signal_logs,ignore_index=True)


# ===============================================================================================
# TRADE_NO NOW REFERS TO ACTUAL EXECUTABLE TRADES,
# NOT RAW MODEL SIGNALS.
# ===============================================================================================

result_df = result_df[result_df["EXECUTABLE_TRADES"] >= TRADE_NO].copy()

# ===============================================================================================
# SORT RESULTS
#
# Primary:
#     WIN RATE
#
# Secondary:
#     PROFIT FACTOR
# ===============================================================================================

result_df = result_df.sort_values(by=["WIN_RATE","PROFIT_FACTOR"],ascending=[False,False])

# ===============================================================================================
# FINAL OUTPUT
# ===============================================================================================

print(f"\n\nSYMBOL: {SYMBOL}")

print("FORWARD TEST START:",df["Date"].min())

print("FORWARD TEST END:",df["Date"].max(),"\n")

print(f"LIVE SPREAD LIMIT: {SPREAD_PERCENT:.2%}")

print("\nTOP THRESHOLD RESULTS:\n")
print(result_df.head(10).to_string(index=False))



# ===============================================================================================
# OPTIONAL: SAVE THRESHOLD RESULTS
# ===============================================================================================

RESULT_FILE = (f"{CSV_FOLDER}/THRESHOLD_RESULTS_BINARY_{SYMBOL}_{CD_TIME}.csv")

result_df.to_csv(RESULT_FILE,index=False)

# ============================================================
# SAVE SIGNAL LOG — BEST THRESHOLD ± 0.01
# ============================================================

min_saving_threshold = result_df["THRESHOLD"].iloc[0] - 0.01
max_saving_threshold = result_df["THRESHOLD"].iloc[0] + 0.01

signal_log_to_save = signal_log_df[
    (signal_log_df["THRESHOLD"] >= min_saving_threshold) &
    (signal_log_df["THRESHOLD"] <= max_saving_threshold)].copy()

SIGNAL_LOG_FILE = (f"{CSV_FOLDER}/SIGNAL_TIME_LOG_BINARY_{SYMBOL}_{CD_TIME}.csv")
signal_log_to_save.to_csv(SIGNAL_LOG_FILE,index=False)


# ===============================================================================================
# SIGNAL TIME INFORMATION
# ===============================================================================================

print("\n\nSIGNAL LOG SAMPLE:\n")

print(signal_log_to_save.head(20).to_string(index=False))

print("\n[[GOOD]] Results saved to:")

print(RESULT_FILE)

print("\n[[GOOD]] Signal time log saved to:")
print(SIGNAL_LOG_FILE)
