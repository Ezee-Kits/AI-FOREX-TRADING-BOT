
#===============================================================================================
# FIND THRESHOLD
#===============================================================================================


import json
import os
import joblib
import json
import numpy as np
import pandas as pd
from func import apply_features,normalize_symbol


# 'AUDCAD',AUDJPY ,'AUDUSD', BTCUSD,CADJPY,ETHUSD,EURJPY, "EURUSD",GBPJPY,
#  "GBPUSD",NZDCAD ,'NZDUSD','USDCAD','USDCHF',"USDJPY", 'USDSEK",'XAGUSD',"XAUUSD


SYMBOL = "EURUSD"
TRADE_NO = 225
CD_TIME = '10M'

risk_percent = 1

tp_mult = 1.75
sl_mult = 1.75


HIGH_TARGETS = ['THL_3H']
weights = {'THL_3H' : 1.0}


BASE_PATH = "/content/drive/MyDrive/FOREX TRADING/ALL_MODELS"

def load_models(model_type,CD_TIME, symbol):
    symbol = normalize_symbol(symbol = symbol)
    models_dict = {}

    for target in HIGH_TARGETS:
        file_path = f"{BASE_PATH}/HL_{model_type}_{target}_{CD_TIME}_{symbol}_model.pkl"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model not found: {file_path}")

        bundle = joblib.load(file_path)

        models_dict[target] = {
            "model": bundle["model"],
            "features": bundle["features"]
        }

        print(f" \n [[GOOD]] Loaded {model_type} model for {target}")

    return models_dict


# LOAD ALL MODELS HERE
HELPER_MODELS = load_models("MAIN", CD_TIME, SYMBOL)

with open("/content/drive/MyDrive/FOREX TRADING/CSV_FILES/SYMBOL_INFO.json") as f:
    SYMBOL_INFO = json.load(f)

info = SYMBOL_INFO[SYMBOL]

balance = info["balance"]
equity = info["equity"]
currency = info["currency"]
leverage = info["leverage"]

pip_size = info["pip_size"]
pip_value_per_lot = info["pip_value_per_lot"]
spread = info["spread"]
min_gap = info["min_gap"]

vol_info = info["vol_info"]

min_lot = vol_info["min"]
max_lot = vol_info["max"]
lot_step = vol_info["step"]




MODEL_FOLDER = ("/content/drive/MyDrive/FOREX TRADING/ALL_MODELS")
CSV_FOLDER = ("/content/drive/MyDrive/FOREX TRADING/CSV_FILES")
IMPORTANCE_FOLDER = ("/content/drive/MyDrive/FOREX TRADING/SHAP RESULT")

# MODEL_FOLDER = 'ALL_MODELS'
# CSV_FOLDER = 'CSV_FILES'
# IMPORTANCE_FOLDER = 'SHAP RESULTS'


POINTS_FILE = os.path.join(CSV_FOLDER,"symbol_points.json")
MT5_SYMBOL = SYMBOL + "c"

print(f"CURRENTLY RUNNING SYMBOL: {SYMBOL}")
print(f"MT5 SYMBOL: {MT5_SYMBOL}")

with open(POINTS_FILE, "r", encoding="utf-8") as file:
    SYMBOL_POINTS = json.load(file)

if MT5_SYMBOL not in SYMBOL_POINTS:
    raise ValueError(f"Point information not found for {MT5_SYMBOL} in {POINTS_FILE}")


POINT = SYMBOL_POINTS[MT5_SYMBOL]["point"]
print(f"POINT: {POINT}")

data = pd.read_csv(f'/content/drive/MyDrive/FOREX TRADING/CSV_FILES/MT5_{CD_TIME}_BT_{SYMBOL}_Exchange_Rate_Dataset.csv')
print('\t DATASET LOADED SUCCESSFULLY \n STARTING APPLY FEATURES ')

df = apply_features(df=data,point=POINT)

# # -----------------------------------------------------------------------
# # FORWARD TEST: JANUARY → JUNE
# df = df[(df['Date'] >= '2026-01-01') & (df['Date'] < '2026-07-01')].copy()
# print("FORWARD TEST START:", df['Date'].min())
# print("FORWARD TEST END:", df['Date'].max())
# # ----------------------------------------------------------------------

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
# df.drop(columns=['Date'], inplace=True)

print('DF INFO:', df.shape)
print('\t APPLY FEATURE SUCCESSFULLY ')



working_df = df[['High','Low','Open','Close','ATR']].copy()

for target in HIGH_TARGETS:

    print(f'\n [[ CURRENTLY PREDICTING TARGET : {target} ]]')

    help_model = HELPER_MODELS[target]["model"]
    help_cols = HELPER_MODELS[target]["features"]

    X_help = df[help_cols]

    # Safety check
    assert set(help_cols) == set(X_help.columns)
    help_proba = help_model.predict_proba(X_help)

    up_prob = help_proba[:,1]
    down_prob = help_proba[:,0]

    working_df[f'{target}_UP'] = up_prob
    working_df[f'{target}_DN'] = down_prob


weight_list = [weights[t] for t in HIGH_TARGETS]
total_weight = sum(weight_list)

working_df['UP_AVG'] = sum(working_df[f"{t}_UP"] * weights[t] for t in HIGH_TARGETS) / total_weight
working_df['DN_AVG'] = sum(working_df[f"{t}_DN"] * weights[t] for t in HIGH_TARGETS) / total_weight

working_df['ATR_PIPS'] = working_df['ATR']/pip_size
working_df['SL_PIPS'] = working_df['ATR_PIPS']*sl_mult
working_df['TP_PIPS'] = working_df['ATR_PIPS']*tp_mult

working_df['NEXT_ENTRY'] = working_df['Open'].shift(-1)
working_df['ASK_PRICE'] = working_df['NEXT_ENTRY']+ spread
working_df['BID_PRICE'] = working_df['NEXT_ENTRY']

working_df['ENTRY_BUY'] = working_df['ASK_PRICE']
working_df['ENTRY_SELL'] = working_df['BID_PRICE']

# BUY trade SL/TP
working_df['SL_BUY'] = working_df['ENTRY_BUY'] - (working_df['SL_PIPS']*pip_size)
working_df['TP_BUY'] = working_df['ENTRY_BUY'] + (working_df['TP_PIPS']*pip_size)

# SELL trade SL/TP
working_df['SL_SELL'] = working_df['ENTRY_SELL'] + (working_df['SL_PIPS']*pip_size)
working_df['TP_SELL'] = working_df['ENTRY_SELL'] - (working_df['TP_PIPS']*pip_size)



def calc_lot_size(
        balance,risk_percent,sl_pips,pip_value_per_lot,min_lot,max_lot):

    risk_amount = balance * (risk_percent / 100)

    lot_cal = risk_amount / (sl_pips * pip_value_per_lot)
    lot = max(min_lot, min(lot_cal, max_lot))

    return lot


def normalize_lot(lot, vol_min, vol_max, vol_step):

    lot = max(vol_min, min(lot, vol_max))

    lot = np.floor(lot / vol_step) * vol_step

    return round(lot, 2)


# ============================================================
# BACKTEST ONE THRESHOLD
# ============================================================

def run_backtest(working_df, threshold):

    # ========================================================
    # SIGNALS
    # ========================================================

    signals = np.zeros(len(working_df), dtype=np.int8)

    up = working_df['UP_AVG'].to_numpy()
    dn = working_df['DN_AVG'].to_numpy()

    signals[up >= threshold] = 1
    signals[dn >= threshold] = -1

    highs = working_df['High'].to_numpy()
    lows = working_df['Low'].to_numpy()

    tp_buy = working_df['TP_BUY'].to_numpy()
    sl_buy = working_df['SL_BUY'].to_numpy()

    tp_sell = working_df['TP_SELL'].to_numpy()
    sl_sell = working_df['SL_SELL'].to_numpy()


    # ========================================================
    # ACCOUNT VARIABLES
    # ========================================================

    # IMPORTANT:
    # Every threshold starts from the SAME balance.
    threshold_balance = balance

    starting_balance = balance

    wins = 0
    losses = 0

    total_gross_profit = 0.0
    total_gross_loss = 0.0

    equity_curve = []

    trade_returns = []


    # ========================================================
    # BACKTEST LOOP
    # ========================================================

    i = 0
    n = len(signals)

    while i < n - 1:

        signal = signals[i]

        if signal == 0:
            i += 1
            continue


        # ====================================================
        # LOT SIZE
        # ====================================================

        sl_pips = working_df['SL_PIPS'].iloc[i]

        lot_size = calc_lot_size(threshold_balance,risk_percent,sl_pips,pip_value_per_lot,min_lot=min_lot,max_lot=max_lot)
        lot_size = normalize_lot(lot_size, vol_info["min"], vol_info["max"], vol_info["step"])

        balance_before_trade = threshold_balance


        # ====================================================
        # BUY
        # ====================================================

        if signal == 1:

            sl = sl_buy[i]
            tp = tp_buy[i]

            entry = working_df['ENTRY_BUY'].iloc[i]

            j = i + 1

            while j < n:

                # -----------------------------
                # STOP LOSS
                # -----------------------------

                if lows[j] <= sl:

                    loss_amount = ( ((entry - sl) / pip_size) * pip_value_per_lot  * lot_size )

                    threshold_balance -= loss_amount
                    total_gross_loss += loss_amount

                    losses += 1

                    trade_return = ( (threshold_balance - balance_before_trade)  / balance_before_trade )
                    trade_returns.append(trade_return)
                    equity_curve.append(threshold_balance)

                    i = j

                    break


                # -----------------------------
                # TAKE PROFIT
                # -----------------------------

                if highs[j] >= tp:

                    profit_amount = (((tp - entry) / pip_size) * pip_value_per_lot * lot_size )

                    threshold_balance += profit_amount

                    total_gross_profit += profit_amount

                    wins += 1

                    trade_return = ((threshold_balance - balance_before_trade) / balance_before_trade )

                    trade_returns.append(trade_return)

                    equity_curve.append(threshold_balance)

                    i = j

                    break

                j += 1


            if j == n:
                i += 1


        # ====================================================
        # SELL
        # ====================================================

        else:

            sl = sl_sell[i]
            tp = tp_sell[i]

            entry = working_df['ENTRY_SELL'].iloc[i]

            j = i + 1

            while j < n:

                # -----------------------------
                # STOP LOSS
                # -----------------------------

                if highs[j] >= sl:

                    loss_amount = (((sl - entry) / pip_size) * pip_value_per_lot * lot_size  )
                    threshold_balance -= loss_amount
                    total_gross_loss += loss_amount
                    losses += 1

                    trade_return = ((threshold_balance - balance_before_trade) / balance_before_trade )
                    trade_returns.append(trade_return)
                    equity_curve.append(threshold_balance)

                    i = j

                    break


                # -----------------------------
                # TAKE PROFIT
                # -----------------------------

                if lows[j] <= tp:

                    profit_amount = (((entry - tp) / pip_size)  * pip_value_per_lot * lot_size )

                    threshold_balance += profit_amount

                    total_gross_profit += profit_amount

                    wins += 1

                    trade_return = ( (threshold_balance - balance_before_trade) / balance_before_trade )
                    trade_returns.append(trade_return)
                    equity_curve.append(threshold_balance)

                    i = j

                    break

                j += 1


            if j == n:
                i += 1


    # ========================================================
    # TOTAL TRADES
    # ========================================================

    total_trades = wins + losses


    # ========================================================
    # WIN RATE
    # ========================================================

    if total_trades > 0:

        win_rate = wins / total_trades

    else:

        win_rate = 0.0


    # ========================================================
    # PROFIT FACTOR
    # ========================================================

    if total_gross_loss > 0:
        profit_factor = (total_gross_profit / total_gross_loss )
    else:
        profit_factor = float("inf")


    # ========================================================
    # EXPECTANCY
    # ========================================================

    if wins > 0:
        avg_win = ( total_gross_profit / wins )
    else:
        avg_win = 0.0


    if losses > 0:
        avg_loss = (total_gross_loss / losses )
    else:
        avg_loss = 0.0


    win_probability = (wins / total_trades if total_trades > 0 else 0)

    loss_probability = (  losses / total_trades  if total_trades > 0 else 0 )

    expectancy = ( (win_probability * avg_win) - (loss_probability * avg_loss) )


    # ========================================================
    # MAXIMUM DRAWDOWN
    # ========================================================

    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve)
        running_peak = equity_series.cummax()
        drawdowns = (  (running_peak - equity_series) / running_peak )

        max_drawdown = drawdowns.max()

    else:
        max_drawdown = 0.0


    # ========================================================
    # TRADE SHARPE
    # ========================================================

    if len(trade_returns) > 1:

        trade_returns_array = np.array(trade_returns)

        returns_std = trade_returns_array.std(ddof=1)

        if returns_std > 0:
            trade_sharpe = ( trade_returns_array.mean()  /  returns_std ) * np.sqrt(252)
        else:
            trade_sharpe = 0.0

    else:
        trade_sharpe = 0.0


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "THRESHOLD":round(threshold, 3),
        "TRADES": total_trades,
        "WINS":wins,
        "LOSSES": losses,

        "WIN_RATE": round(win_rate * 100, 2),
        "PROFIT_FACTOR":round(profit_factor, 2) if np.isfinite(profit_factor) else np.inf,
        "EXPECTANCY": round(expectancy, 2),
        "MAX_DRAWDOWN": round(max_drawdown * 100, 2),
        "TRADE_SHARPE":round(trade_sharpe, 2),

        "STARTING_BALANCE":round(starting_balance, 2),
        "FINAL_BALANCE":round(threshold_balance, 2),
        "GROSS_PROFIT":round(total_gross_profit, 2),
        "GROSS_LOSS":round(total_gross_loss, 2)
    }


results = []
thresholds = sorted(set(pd.concat([working_df['UP_AVG'],working_df['DN_AVG']]).round(3).dropna()))
print('THRESHOL LENGHT >>',len(thresholds))

for threshold in thresholds:
    print(f'CURRENTLY ON {threshold} THRESHOLD >> {thresholds.index(threshold)+1}/{len(thresholds)}')

    result = run_backtest(working_df,threshold)
    results.append(result)

result_df = pd.DataFrame(results)
result_df = result_df[result_df['TRADES'] >= TRADE_NO]

result_df = result_df.sort_values(by='WIN_RATE',ascending=False)

print(f'\n \n SYMBOL: {SYMBOL}')
print("FORWARD TEST START:", df['Date'].min())
print("FORWARD TEST END:", df['Date'].max(),'\n')

print(result_df.head(10).to_string(index=False))