import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global states
pending_reversal_long = False
pending_reversal_short = False
entry_order_id = None
current_direction = None

# Configuration
EXCHANGE = 'bybit'
SYMBOL = 'SOL/USDT:USDT'  # Corrected USDT perpetual symbol for SOL
TIMEFRAME = '30m'
EXPOSURE_MULTIPLIER = 3  # Multiplier sul saldo totale per esposizione notionale (es. 2.5x = 250% del saldo)
SLACK = 1.0
DIST_MIN = 0.02
STOCH_LENGTH = 30
SMOOTH_K = 27
SMOOTH_D = 26
EMA_LENGTH = 20
USE_EMA_FILTER = True
TP_PERCENT = 2.1  # Take Profit percentage
RSI_LENGTH = 30  # Hardcoded since no filter

# Initialize exchange
exchange = ccxt.bybit({
    'apiKey': '................',  # Replace with your API key
    'secret': '..................',   # Replace with your secret
    'sandbox': False,          # Set to True for testnet
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # For USDT perpetual futures
    },
})

# Function to fetch USDT balance
def get_usdt_balance():
    try:
        balance = exchange.fetch_balance()
        usdt_free = balance['USDT']['free']
        logger.info(f"USDT free balance: {usdt_free}")
        return usdt_free
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")
        return 0.0

# Function to get best bid and ask from order book
def get_best_bid_ask():
    try:
        orderbook = exchange.fetch_order_book(SYMBOL, limit=1)
        best_bid = orderbook['bids'][0][0] if orderbook['bids'] else None
        best_ask = orderbook['asks'][0][0] if orderbook['asks'] else None
        if best_bid is None or best_ask is None:
            raise ValueError("No bid/ask available")
        logger.info(f"Best bid: {best_bid}, Best ask: {best_ask}")
        return best_bid, best_ask
    except Exception as e:
        logger.error(f"Error fetching order book: {e}")
        return None, None

# Function to compute Stochastic RSI
def compute_stoch_rsi(close, rsi_len, stoch_len):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    low_rsi = rsi.rolling(window=stoch_len).min()
    high_rsi = rsi.rolling(window=stoch_len).max()
    stoch_rsi = 100 * (rsi - low_rsi) / (high_rsi - low_rsi)
    return stoch_rsi

# Function to get K and D
def get_stoch_tuple(close, rsi_len, stoch_len, k_len, d_len):
    stoch_rsi = compute_stoch_rsi(close, rsi_len, stoch_len)
    k = stoch_rsi.rolling(window=k_len).mean()
    d = k.rolling(window=d_len).mean()
    return stoch_rsi, k, d

# Function to compute EMA
def compute_ema(series, period):
    return series.ewm(span=period).mean()

# Function to get current position size (positive for long, negative for short) + entry price and side
def get_position_size():
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if pos['symbol'] == SYMBOL and pos['contracts'] > 0:
                size = pos['contracts'] if pos['side'] == 'long' else -pos['contracts']
                entry_price = float(pos['entryPrice']) if 'entryPrice' in pos else None
                return size, entry_price, pos['side']
        return 0, None, None
    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        return 0, None, None

# Function to place TP order
def place_tp(entry_price, direction):
    global current_direction
    try:
        position_size, _, _ = get_position_size()
        if abs(position_size) == 0:
            logger.warning("No position to set TP")
            return False
        quantity = abs(position_size)
        quantity = exchange.amount_to_precision(SYMBOL, quantity)
        if direction == 'long':
            tp_price = entry_price * (1 + TP_PERCENT / 100)
            tp_price = exchange.price_to_precision(SYMBOL, tp_price)
            order = exchange.create_limit_sell_order(SYMBOL, quantity, tp_price, params={'reduceOnly': True})
            logger.info(f"Placed TP for long: sell {quantity} at {tp_price} ({TP_PERCENT}% profit)")
        elif direction == 'short':
            tp_price = entry_price * (1 - TP_PERCENT / 100)
            tp_price = exchange.price_to_precision(SYMBOL, tp_price)
            order = exchange.create_limit_buy_order(SYMBOL, quantity, tp_price, params={'reduceOnly': True})
            logger.info(f"Placed TP for short: buy {quantity} at {tp_price} ({TP_PERCENT}% profit)")
        return True
    except Exception as e:
        logger.error(f"Error placing TP: {e}")
        return False

# Function to close position using limit order
def close_position():
    try:
        position_size, _, _ = get_position_size()
        if position_size == 0:
            return True
        best_bid, best_ask = get_best_bid_ask()
        if best_bid is None or best_ask is None:
            logger.error("Could not get bid/ask for close")
            return False
        quantity = abs(position_size)
        quantity = exchange.amount_to_precision(SYMBOL, quantity)
        if position_size > 0:
            # Close long: sell limit at best bid
            order = exchange.create_limit_sell_order(SYMBOL, quantity, best_bid)
            logger.info(f"Closed long position with limit sell: {quantity} SOL at {best_bid} USDT")
        elif position_size < 0:
            # Close short: buy limit at best ask
            order = exchange.create_limit_buy_order(SYMBOL, quantity, best_ask)
            logger.info(f"Closed short position with limit buy: {quantity} SOL at {best_ask} USDT")
        return order
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return None

# Function to enter long using limit order
def enter_long():
    try:
        usdt_balance = get_usdt_balance()
        position_value_usdt = usdt_balance * EXPOSURE_MULTIPLIER
        ticker = exchange.fetch_ticker(SYMBOL)
        price = ticker['last']
        quantity = position_value_usdt / price  # Quantity in SOL
        quantity = exchange.amount_to_precision(SYMBOL, quantity)
        best_bid, best_ask = get_best_bid_ask()
        if best_ask is None:
            logger.error("Could not get best ask for long entry")
            return None
        # Enter long: buy limit at best ask
        order = exchange.create_limit_buy_order(SYMBOL, quantity, best_ask)
        logger.info(f"Entered long with limit buy: {quantity} SOL at {best_ask} USDT, notional {position_value_usdt:.2f} USDT ({EXPOSURE_MULTIPLIER}x exposure)")
        return order
    except Exception as e:
        logger.error(f"Error entering long: {e}")
        return None

# Function to enter short using limit order
def enter_short():
    try:
        usdt_balance = get_usdt_balance()
        position_value_usdt = usdt_balance * EXPOSURE_MULTIPLIER
        ticker = exchange.fetch_ticker(SYMBOL)
        price = ticker['last']
        quantity = position_value_usdt / price  # Quantity in SOL
        quantity = exchange.amount_to_precision(SYMBOL, quantity)
        best_bid, best_ask = get_best_bid_ask()
        if best_bid is None:
            logger.error("Could not get best bid for short entry")
            return None
        # Enter short: sell limit at best bid
        order = exchange.create_limit_sell_order(SYMBOL, quantity, best_bid)
        logger.info(f"Entered short with limit sell: {quantity} SOL at {best_bid} USDT, notional {position_value_usdt:.2f} USDT ({EXPOSURE_MULTIPLIER}x exposure)")
        return order
    except Exception as e:
        logger.error(f"Error entering short: {e}")
        return None

# Function to calculate sleep time to next 30m candle
def time_to_next_candle(current_candle_time):
    now = datetime.utcnow()
    next_candle = current_candle_time + timedelta(minutes=30)
    time_diff = (next_candle - now).total_seconds()
    return max(0, time_diff)

# Main loop
def main():
    global pending_reversal_long, pending_reversal_short, entry_order_id, current_direction
    # Check connection and balance at startup
    logger.info("Starting bot...")
    try:
        # Set position mode to one-way to avoid hedge mode issues
        exchange.set_position_mode(False, SYMBOL)
        logger.info("Set position mode to one-way")
    except Exception as e:
        logger.warning(f"Could not set position mode (may already be set): {e}")
    
    usdt_balance = get_usdt_balance()
    if usdt_balance > 0:
        logger.info(f"Successfully connected to Bybit. Available USDT: {usdt_balance}")
    else:
        logger.error("Failed to connect or fetch balance. Exiting.")
        return

    # Initialize current_direction based on existing position at startup
    position_size, _, _ = get_position_size()
    if position_size > 0:
        current_direction = 'long'
        logger.info("Detected existing long position at startup - assuming TP already set")
    elif position_size < 0:
        current_direction = 'short'
        logger.info("Detected existing short position at startup - assuming TP already set")
    else:
        current_direction = None

    pending_reversal_long = False
    pending_reversal_short = False
    entry_order_id = None

    last_candle_time = None

    while True:
        try:
            # Fetch latest OHLCV
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            current_candle_time = df.index[-1]
            new_candle = last_candle_time is None or current_candle_time > last_candle_time

            if new_candle:
                logger.info(f"New 30m candle closed at {current_candle_time}")
                last_candle_time = current_candle_time

                # Compute indicators
                close = df['close']
                _, k, d = get_stoch_tuple(close, RSI_LENGTH, STOCH_LENGTH, SMOOTH_K, SMOOTH_D)
                ema = compute_ema(close, EMA_LENGTH)

                current_k = k.iloc[-1]
                current_d = d.iloc[-1]
                prev_k_val = k.iloc[-2] if len(k) > 1 else np.nan
                prev_d_val = d.iloc[-2] if len(d) > 1 else np.nan
                current_ema = ema.iloc[-1]
                current_close = close.iloc[-1]

                prev_dist = abs(prev_k_val - prev_d_val) if not np.isnan(prev_k_val) and not np.isnan(prev_d_val) else 0

                # Conditions
                long_cross = current_k > current_d and prev_k_val <= (prev_d_val + SLACK)
                short_cross = current_k < current_d and prev_k_val >= (prev_d_val - SLACK)

                # Filters (only EMA)
                ema_long_ok = not USE_EMA_FILTER or current_close > current_ema
                ema_short_ok = not USE_EMA_FILTER or current_close < current_ema

                position_size, _, _ = get_position_size()

                # Debug logging
                logger.info(f"Indicators: K={current_k:.2f}, D={current_d:.2f}, prevK={prev_k_val:.2f}, prevD={prev_d_val:.2f}, Close={current_close:.4f}, EMA={current_ema:.4f}")
                logger.info(f"Crosses: Long={long_cross}, Short={short_cross}, Dist={prev_dist:.2f} >= {DIST_MIN}")
                logger.info(f"Filters: EMA Long OK={ema_long_ok}, EMA Short OK={ema_short_ok}")
                logger.info(f"Position size: {position_size}")

                # Close conditions for reversal
                closed = False
                if position_size > 0 and short_cross:
                    logger.info("Closing long due to short cross")
                    close_position()
                    closed = True
                    pending_reversal_short = True
                if position_size < 0 and long_cross:
                    logger.info("Closing short due to long cross")
                    close_position()
                    closed = True
                    pending_reversal_long = True

                # Re-fetch position_size after close (to avoid double orders in reversal)
                if closed:
                    time.sleep(2)  # Breve pausa per sincronizzazione
                    position_size, _, _ = get_position_size()

                # Entry conditions (normal signals or reversals) - use updated position_size
                if long_cross and prev_dist >= DIST_MIN and position_size <= 0 and ema_long_ok:
                    logger.info("Long entry signal triggered")
                    open_order = enter_long()
                    if open_order:
                        entry_order_id = open_order['id']
                        current_direction = 'long'
                elif short_cross and prev_dist >= DIST_MIN and position_size >= 0 and ema_short_ok:
                    logger.info("Short entry signal triggered")
                    open_order = enter_short()
                    if open_order:
                        entry_order_id = open_order['id']
                        current_direction = 'short'

            # Check for pending reversals (every loop) - log position only if pending
            if pending_reversal_long or pending_reversal_short:
                position_size, _, _ = get_position_size()
                logger.info(f"Position size for reversal check: {position_size}")
                if pending_reversal_long and position_size <= 0:
                    logger.info("Executing pending long reversal")
                    open_order = enter_long()
                    if open_order:
                        entry_order_id = open_order['id']
                        current_direction = 'long'
                        # Quick check post-entry
                        time.sleep(1)
                        pos_size, entry_price_quick, pos_side = get_position_size()
                        if pos_size > 0 and pos_side == 'long':
                            logger.info(f"Reversal long filled quickly: size={pos_size}")
                            # Prova TP immediato se position ok
                            ticker = exchange.fetch_ticker(SYMBOL)
                            avg_price = float(open_order.get('average', entry_price_quick or ticker['last']))
                            place_tp(avg_price, 'long')
                            entry_order_id = None  # Reset se già filled
                        pending_reversal_long = False
                elif pending_reversal_short and position_size >= 0:
                    logger.info("Executing pending short reversal")
                    open_order = enter_short()
                    if open_order:
                        entry_order_id = open_order['id']
                        current_direction = 'short'
                        time.sleep(1)
                        pos_size, entry_price_quick, pos_side = get_position_size()
                        if pos_size < 0 and pos_side == 'short':
                            logger.info(f"Reversal short filled quickly: size={pos_size}")
                            ticker = exchange.fetch_ticker(SYMBOL)
                            avg_price = float(open_order.get('average', entry_price_quick or ticker['last']))
                            place_tp(avg_price, 'short')
                            entry_order_id = None
                        pending_reversal_short = False

            # Check entry order fill and place TP (every loop) - con fallback su position
            if entry_order_id:
                order_filled = False
                entry_price_from_order = None
                try:
                    # Prova fetch_order con acknowledged=True per sopprimere warning
                    order = exchange.fetch_order(entry_order_id, SYMBOL, params={'acknowledged': True})
                    if order['status'] == 'closed' and float(order['filled']) > 0:
                        entry_price_from_order = float(order['average'])
                        logger.info(f"Entry order filled at {entry_price_from_order}")
                        order_filled = True
                    elif order['status'] in ['canceled', 'rejected']:
                        logger.error(f"Entry order {order['status']}: {order.get('info', 'No info')}")
                        entry_order_id = None
                        continue  # Skip TP
                except Exception as e:
                    logger.warning(f"fetch_order failed (likely old order): {e}. Falling back to position check.")

                if not order_filled:
                    # Fallback: Check position
                    position_size, entry_price_from_pos, pos_side = get_position_size()
                    expected_side = 'long' if pending_reversal_long or (current_direction == 'long') else 'short'
                    if abs(position_size) > 0 and pos_side == expected_side:
                        logger.info(f"Position active (fallback): size={position_size}, entry_price={entry_price_from_pos}, side={pos_side}")
                        entry_price_from_order = entry_price_from_pos  # Usa questo
                        order_filled = True
                    else:
                        logger.info(f"No active position for fallback (size={position_size}), skipping TP check")
                        continue

                if order_filled and entry_price_from_order:
                    # Retry loop per TP
                    max_retries = 3
                    tp_success = False
                    for attempt in range(max_retries):
                        if place_tp(entry_price_from_order, current_direction):
                            logger.info(f"TP placed successfully on attempt {attempt + 1}")
                            tp_success = True
                            entry_order_id = None  # Reset solo dopo TP success
                            break
                        else:
                            logger.warning(f"TP attempt {attempt + 1} failed, retrying...")
                            time.sleep(2)
                    if not tp_success:
                        logger.error("Failed to place TP after retries - manual check needed!")
                else:
                    entry_order_id = None  # Reset se non filled

            # Adaptive sleep: 30s if pending stuff, else time to next 30m candle
            has_pending = entry_order_id or pending_reversal_long or pending_reversal_short
            if has_pending:
                sleep_time = 30
                logger.debug(f"Pending activity detected - sleeping {sleep_time}s for quick check")
            else:
                sleep_time = time_to_next_candle(current_candle_time)
                if sleep_time > 0:
                    logger.info(f"No pending activity - sleeping {sleep_time:.0f}s until next 30m candle")
                else:
                    sleep_time = 1  # If already past, sleep minimal to refetch
            time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
