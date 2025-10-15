import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
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
DIST_MIN = 0.0
RSI_LENGTH = 30
STOCH_LENGTH = 30
SMOOTH_K = 27
SMOOTH_D = 26
EMA_LENGTH = 20
USE_RSI_FILTER = False
USE_EMA_FILTER = True
TP_PERCENT = 2.2  # Take Profit percentage

# Initialize exchange
exchange = ccxt.bybit({
    'apiKey': 'JWK8qGoHQnBUB7EU0M',  # Replace with your API key
    'secret': 'lFOzErJbHf3bs5uneFXDhDPKQYuUyxbn0VMX',   # Replace with your secret
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

# Function to compute RSI
def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute Stochastic RSI
def compute_stoch_rsi(close, rsi_len, stoch_len):
    rsi = compute_rsi(close, rsi_len)
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

# Function to get current position size (positive for long, negative for short)
def get_position_size():
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if pos['symbol'] == SYMBOL and pos['contracts'] > 0:
                return pos['contracts'] if pos['side'] == 'long' else -pos['contracts']
        return 0
    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        return 0

# Function to place TP order
def place_tp(entry_price, direction):
    global current_direction
    try:
        position_size = get_position_size()
        if position_size == 0:
            logger.warning("No position to set TP")
            return
        quantity = abs(position_size)
        quantity = exchange.amount_to_precision(SYMBOL, quantity)
        if direction == 'long':
            tp_price = entry_price * (1 + TP_PERCENT / 100)
            tp_price = exchange.price_to_precision(SYMBOL, tp_price)
            order = exchange.create_limit_sell_order(SYMBOL, quantity, tp_price, params={'reduceOnly': True})
            logger.info(f"Placed TP for long: sell {quantity} at {tp_price} (2.2% profit)")
        elif direction == 'short':
            tp_price = entry_price * (1 - TP_PERCENT / 100)
            tp_price = exchange.price_to_precision(SYMBOL, tp_price)
            order = exchange.create_limit_buy_order(SYMBOL, quantity, tp_price, params={'reduceOnly': True})
            logger.info(f"Placed TP for short: buy {quantity} at {tp_price} (2.2% profit)")
    except Exception as e:
        logger.error(f"Error placing TP: {e}")

# Function to close position using limit order
def close_position():
    try:
        position_size = get_position_size()
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

    pending_reversal_long = False
    pending_reversal_short = False
    entry_order_id = None
    current_direction = None

    last_candle_time = None

    while True:
        try:
            # Fetch latest OHLCV
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=200)  # Increased to 200 for sufficient indicator history
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            current_candle_time = df.index[-1]
            if last_candle_time is None or current_candle_time > last_candle_time:
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
                current_rsi = compute_rsi(close, RSI_LENGTH).iloc[-1]
                current_ema = ema.iloc[-1]
                current_close = close.iloc[-1]

                prev_dist = abs(prev_k_val - prev_d_val) if not np.isnan(prev_k_val) and not np.isnan(prev_d_val) else 0

                # Conditions
                long_cross = current_k > current_d and prev_k_val <= (prev_d_val + SLACK)
                short_cross = current_k < current_d and prev_k_val >= (prev_d_val - SLACK)

                # Filters
                rsi_long_ok = not USE_RSI_FILTER or current_rsi < 30
                rsi_short_ok = not USE_RSI_FILTER or current_rsi > 70
                ema_long_ok = not USE_EMA_FILTER or current_close > current_ema
                ema_short_ok = not USE_EMA_FILTER or current_close < current_ema

                position_size = get_position_size()

                # Debug logging
                logger.info(f"Indicators: K={current_k:.2f}, D={current_d:.2f}, prevK={prev_k_val:.2f}, prevD={prev_d_val:.2f}, RSI={current_rsi:.2f}, Close={current_close:.4f}, EMA={current_ema:.4f}")
                logger.info(f"Crosses: Long={long_cross}, Short={short_cross}, Dist={prev_dist:.2f} >= {DIST_MIN}")
                logger.info(f"Filters: RSI Long OK={rsi_long_ok}, RSI Short OK={rsi_short_ok}, EMA Long OK={ema_long_ok}, EMA Short OK={ema_short_ok}")
                logger.info(f"Position size: {position_size}")

                # Close conditions for reversal
                if position_size > 0 and short_cross:
                    logger.info("Closing long due to short cross")
                    close_position()
                    pending_reversal_short = True
                if position_size < 0 and long_cross:
                    logger.info("Closing short due to long cross")
                    close_position()
                    pending_reversal_long = True

                # Entry conditions (normal signals)
                if long_cross and prev_dist >= DIST_MIN and position_size <= 0 and ema_long_ok and rsi_long_ok:
                    logger.info("Long entry signal triggered")
                    open_order = enter_long()
                    if open_order:
                        entry_order_id = open_order['id']
                        current_direction = 'long'
                elif short_cross and prev_dist >= DIST_MIN and position_size >= 0 and ema_short_ok and rsi_short_ok:
                    logger.info("Short entry signal triggered")
                    open_order = enter_short()
                    if open_order:
                        entry_order_id = open_order['id']
                        current_direction = 'short'

            # Check for pending reversals (every loop)
            position_size = get_position_size()
            if pending_reversal_long and position_size <= 0:
                logger.info("Executing pending long reversal")
                open_order = enter_long()
                if open_order:
                    entry_order_id = open_order['id']
                    current_direction = 'long'
                pending_reversal_long = False
            elif pending_reversal_short and position_size >= 0:
                logger.info("Executing pending short reversal")
                open_order = enter_short()
                if open_order:
                    entry_order_id = open_order['id']
                    current_direction = 'short'
                pending_reversal_short = False

            # Check entry order fill and place TP (every loop)
            if entry_order_id:
                try:
                    order = exchange.fetch_order(entry_order_id, SYMBOL)
                    if order['status'] == 'closed' and float(order['filled']) > 0:
                        entry_price = float(order['average'])
                        logger.info(f"Entry order filled at {entry_price}")
                        place_tp(entry_price, current_direction)
                        entry_order_id = None
                    elif order['status'] in ['canceled', 'rejected']:
                        logger.error(f"Entry order {order['status']}: {order.get('info', 'No info')}")
                        entry_order_id = None
                except Exception as e:
                    logger.error(f"Error checking entry order: {e}")

            # Wait 30 seconds (approx every 30s check)
            time.sleep(30)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
