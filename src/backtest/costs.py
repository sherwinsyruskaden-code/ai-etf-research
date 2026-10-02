import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TradingCostModel:
    """
    Realistic Financial Friction Model for Long-Only Indian ETF Trading:
    1. Groww Brokerage (Equity Delivery): Max(0, Min(0.0005 * Trade_Value, 20.0))
    2. Statutory Taxes & Exchange Fees (NSE Delivery):
       - STT (Securities Transaction Tax): 0.10% on Buy + 0.10% on Sell (0.20% roundtrip)
       - Stamp Duty: 0.015% on Buy
       - Exchange Turnover Charge (NSE): 0.00345% on Buy & Sell
       - SEBI Turnover Fee: 0.0001% on Buy & Sell
       - GST: 18% on (Brokerage + Exchange Txn Fee + SEBI Fee)
    3. Slippage: Fixed 0.05% on Entry + 0.05% on Exit (0.10% roundtrip)
    4. Estimated Bid-Ask Spread: Scaled by ETF Liquidity (Average Daily Traded Value):
       - High Liquidity (ADTV > 10 Cr): 0.05% roundtrip
       - Medium Liquidity (ADTV 1-10 Cr): 0.10% roundtrip
       - Low Liquidity (ADTV < 1 Cr): 0.20% roundtrip
    """

    def __init__(
        self,
        fixed_slippage_pct: float = 0.0005,  # 0.05% per leg (0.10% roundtrip)
        stt_pct: float = 0.0010,              # 0.10% per leg (0.20% roundtrip)
        stamp_duty_pct: float = 0.00015,      # 0.015% on Buy
        exchange_fee_pct: float = 0.0000345,  # 0.00345% per leg
        sebi_fee_pct: float = 0.000001,       # 0.0001% per leg
        gst_pct: float = 0.18                 # 18% on brokerage + txn fees
    ):
        self.fixed_slippage_pct = fixed_slippage_pct
        self.stt_pct = stt_pct
        self.stamp_duty_pct = stamp_duty_pct
        self.exchange_fee_pct = exchange_fee_pct
        self.sebi_fee_pct = sebi_fee_pct
        self.gst_pct = gst_pct

    def get_spread_pct_for_liquidity(self, avg_daily_volume: float) -> float:
        """
        Estimate roundtrip bid-ask spread percentage based on instrument liquidity.
        """
        if avg_daily_volume > 100_000_000:  # > ₹10 Cr
            return 0.0005  # 5 bps roundtrip
        elif avg_daily_volume > 10_000_000:  # ₹1 Cr - ₹10 Cr
            return 0.0010  # 10 bps roundtrip
        else:  # < ₹1 Cr
            return 0.0020  # 20 bps roundtrip

    def calculate_trade_friction(
        self,
        entry_price: float,
        exit_price: float,
        avg_daily_volume: float = 50_000_000
    ) -> Dict[str, float]:
        """
        Calculate pre-cost and post-cost realized returns for a long-only trade.

        Entry fills at: Entry_Open * (1 + slippage + half_spread)
        Exit fills at: Exit_Price * (1 - slippage - half_spread)

        Args:
            entry_price (float): Raw Open price on entry bar (T+1 Open).
            exit_price (float): Raw Exit price on exit bar.
            avg_daily_volume (float): Average daily traded value for liquidity spread estimation.

        Returns:
            Dict[str, float]: Itemized returns and cost breakdown.
        """
        if entry_price <= 0 or exit_price <= 0:
            return {
                "raw_return": 0.0,
                "net_return": 0.0,
                "total_cost_pct": 0.0,
                "effective_entry_price": entry_price,
                "effective_exit_price": exit_price,
            }

        # 1. Raw Return before any costs/slippage/spread
        raw_return = (exit_price - entry_price) / entry_price

        # 2. Bid-Ask Spread & Slippage
        spread_pct = self.get_spread_pct_for_liquidity(avg_daily_volume)
        half_spread = spread_pct / 2.0

        # Effective execution price after entry slippage + half spread (Buy higher)
        effective_entry = entry_price * (1.0 + self.fixed_slippage_pct + half_spread)

        # Effective execution price after exit slippage + half spread (Sell lower)
        effective_exit = exit_price * (1.0 - self.fixed_slippage_pct - half_spread)

        # Price return after slippage & spread
        spread_slippage_return = (effective_exit - effective_entry) / effective_entry

        # 3. Statutory Taxes & Brokerage (applied on effective trade value)
        # Entry Taxes & Fees: STT (0.1%), Stamp Duty (0.015%), Exchange Fee, SEBI Fee, GST
        entry_brokerage = min(20.0, effective_entry * 0.0005)
        entry_stt = effective_entry * self.stt_pct
        entry_stamp = effective_entry * self.stamp_duty_pct
        entry_exch = effective_entry * self.exchange_fee_pct
        entry_sebi = effective_entry * self.sebi_fee_pct
        entry_gst = (entry_brokerage + entry_exch + entry_sebi) * self.gst_pct
        total_entry_fee_pct = (entry_brokerage + entry_stt + entry_stamp + entry_exch + entry_sebi + entry_gst) / effective_entry

        # Exit Taxes & Fees: STT (0.1%), Exchange Fee, SEBI Fee, GST
        exit_brokerage = min(20.0, effective_exit * 0.0005)
        exit_stt = effective_exit * self.stt_pct
        exit_exch = effective_exit * self.exchange_fee_pct
        exit_sebi = effective_exit * self.sebi_fee_pct
        exit_gst = (exit_brokerage + exit_exch + exit_sebi) * self.gst_pct
        total_exit_fee_pct = (exit_brokerage + exit_stt + exit_exch + exit_sebi + exit_gst) / effective_exit

        total_statutory_pct = total_entry_fee_pct + total_exit_fee_pct

        # 4. Net Post-Cost Return
        net_return = spread_slippage_return - total_statutory_pct
        total_cost_pct = raw_return - net_return

        return {
            "raw_return": float(raw_return),
            "net_return": float(net_return),
            "total_cost_pct": float(total_cost_pct),
            "effective_entry_price": float(effective_entry),
            "effective_exit_price": float(effective_exit),
            "statutory_pct": float(total_statutory_pct),
            "spread_slippage_pct": float(self.fixed_slippage_pct * 2 + spread_pct)
        }
