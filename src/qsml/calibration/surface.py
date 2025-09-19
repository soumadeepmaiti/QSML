import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime, date
import logging
from ..pricers.bs import bs_call_price, bs_implied_vol_newton

logger = logging.getLogger(__name__)


@dataclass
class SmileSlice:
    """
    Represents an implied volatility smile for a single expiration.
    
    This is a fundamental data structure that holds option market data
    and implied volatilities for a specific expiration date.
    """
    expiry_date: Union[datetime, date]
    time_to_expiry: float  # In years
    spot: float
    forward: float
    risk_free_rate: float
    dividend_yield: float
    
    # Option data
    strikes: np.ndarray
    calls_mid: Optional[np.ndarray] = None
    puts_mid: Optional[np.ndarray] = None
    calls_bid: Optional[np.ndarray] = None
    calls_ask: Optional[np.ndarray] = None
    puts_bid: Optional[np.ndarray] = None
    puts_ask: Optional[np.ndarray] = None
    
    # Derived data
    moneyness: Optional[np.ndarray] = field(init=False, default=None)
    log_moneyness: Optional[np.ndarray] = field(init=False, default=None)
    implied_vols: Optional[np.ndarray] = field(init=False, default=None)
    
    # Market data quality
    bid_ask_spreads: Optional[np.ndarray] = field(init=False, default=None)
    volumes: Optional[np.ndarray] = None
    open_interest: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Compute derived quantities after initialization."""
        self.moneyness = self.strikes / self.forward
        self.log_moneyness = np.log(self.moneyness)
        
        # Compute implied volatilities if market prices are available
        if self.calls_mid is not None or self.puts_mid is not None:
            self._compute_implied_volatilities()
        
        # Compute bid-ask spreads if available
        if self.calls_bid is not None and self.calls_ask is not None:
            self.bid_ask_spreads = self.calls_ask - self.calls_bid
    
    def _compute_implied_volatilities(self):
        """Compute implied volatilities from market prices."""
        n_strikes = len(self.strikes)
        self.implied_vols = np.full(n_strikes, np.nan)
        
        for i, K in enumerate(self.strikes):
            # Prefer calls for OTM/ATM options, puts for ITM
            if K >= self.forward and self.calls_mid is not None:
                market_price = self.calls_mid[i]
                option_type = "call"
            elif K < self.forward and self.puts_mid is not None:
                market_price = self.puts_mid[i]
                option_type = "put"
            elif self.calls_mid is not None:
                market_price = self.calls_mid[i]
                option_type = "call"
            elif self.puts_mid is not None:
                market_price = self.puts_mid[i]
                option_type = "put"
            else:
                continue
            
            try:
                iv = bs_implied_vol_newton(
                    S=self.spot,
                    K=K,
                    T=self.time_to_expiry,
                    r=self.risk_free_rate,
                    q=self.dividend_yield,
                    option_price=market_price,
                    option_type=option_type
                )
                self.implied_vols[i] = iv
            except Exception as e:
                logger.debug(f"Failed to compute IV for K={K}: {e}")
                continue
    
    def get_atm_vol(self) -> float:
        """Get at-the-money implied volatility."""
        if self.implied_vols is None:
            return np.nan
        
        # Find closest strike to forward
        atm_idx = np.argmin(np.abs(self.strikes - self.forward))
        return self.implied_vols[atm_idx]
    
    def get_vol_at_moneyness(self, target_moneyness: float) -> float:
        """Interpolate implied volatility at specific moneyness."""
        if self.implied_vols is None:
            return np.nan
        
        # Remove NaN values
        valid_mask = ~np.isnan(self.implied_vols)
        if np.sum(valid_mask) < 2:
            return np.nan
        
        valid_moneyness = self.moneyness[valid_mask]
        valid_vols = self.implied_vols[valid_mask]
        
        return np.interp(target_moneyness, valid_moneyness, valid_vols)
    
    def get_vol_at_delta(self, target_delta: float, option_type: str = "call") -> float:
        """Get implied volatility for a given delta level."""
        if self.implied_vols is None:
            return np.nan
        
        from ..pricers.bs import bs_delta
        
        deltas = []
        valid_vols = []
        
        for i, (K, vol) in enumerate(zip(self.strikes, self.implied_vols)):
            if np.isnan(vol):
                continue
            
            try:
                delta = bs_delta(
                    S=self.spot,
                    K=K,
                    T=self.time_to_expiry,
                    r=self.risk_free_rate,
                    q=self.dividend_yield,
                    sigma=vol,
                    option_type=option_type
                )
                deltas.append(delta)
                valid_vols.append(vol)
            except:
                continue
        
        if len(deltas) < 2:
            return np.nan
        
        deltas = np.array(deltas)
        valid_vols = np.array(valid_vols)
        
        # Sort by delta
        sort_idx = np.argsort(deltas)
        deltas = deltas[sort_idx]
        valid_vols = valid_vols[sort_idx]
        
        return np.interp(target_delta, deltas, valid_vols)
    
    def filter_liquid_options(self, min_volume: float = 0, min_oi: float = 0) -> 'SmileSlice':
        """Filter options based on liquidity criteria."""
        if self.volumes is None and self.open_interest is None:
            return self
        
        # Create liquidity mask
        mask = np.ones(len(self.strikes), dtype=bool)
        
        if self.volumes is not None:
            mask &= (self.volumes >= min_volume)
        
        if self.open_interest is not None:
            mask &= (self.open_interest >= min_oi)
        
        return self._filter_by_mask(mask)
    
    def filter_by_moneyness(self, min_moneyness: float = 0.8, max_moneyness: float = 1.2) -> 'SmileSlice':
        """Filter options by moneyness range."""
        mask = (self.moneyness >= min_moneyness) & (self.moneyness <= max_moneyness)
        return self._filter_by_mask(mask)
    
    def filter_valid_ivs(self) -> 'SmileSlice':
        """Filter out options with invalid implied volatilities."""
        if self.implied_vols is None:
            return self
        
        mask = ~np.isnan(self.implied_vols) & (self.implied_vols > 0) & (self.implied_vols < 5.0)
        return self._filter_by_mask(mask)
    
    def _filter_by_mask(self, mask: np.ndarray) -> 'SmileSlice':
        """Apply a boolean mask to filter all arrays."""
        filtered = SmileSlice(
            expiry_date=self.expiry_date,
            time_to_expiry=self.time_to_expiry,
            spot=self.spot,
            forward=self.forward,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            strikes=self.strikes[mask],
            calls_mid=self.calls_mid[mask] if self.calls_mid is not None else None,
            puts_mid=self.puts_mid[mask] if self.puts_mid is not None else None,
            calls_bid=self.calls_bid[mask] if self.calls_bid is not None else None,
            calls_ask=self.calls_ask[mask] if self.calls_ask is not None else None,
            puts_bid=self.puts_bid[mask] if self.puts_bid is not None else None,
            puts_ask=self.puts_ask[mask] if self.puts_ask is not None else None,
            volumes=self.volumes[mask] if self.volumes is not None else None,
            open_interest=self.open_interest[mask] if self.open_interest is not None else None
        )
        
        return filtered


@dataclass 
class VolatilitySurface:
    """
    Represents a complete implied volatility surface across strikes and expiries.
    
    This is the main data structure for managing option market data and
    provides functionality for surface interpolation, arbitrage checking,
    and data validation.
    """
    valuation_date: Union[datetime, date]
    spot: float
    risk_free_rate: float  # Can be curve later
    dividend_yield: float  # Can be curve later
    
    # Collection of smile slices
    slices: Dict[Union[datetime, date, str], SmileSlice] = field(default_factory=dict)
    
    # Surface metadata
    underlying_symbol: Optional[str] = None
    currency: str = "USD"
    
    def add_slice(self, slice_obj: SmileSlice, key: Optional[str] = None):
        """Add a smile slice to the surface."""
        if key is None:
            key = slice_obj.expiry_date
        
        self.slices[key] = slice_obj
    
    def get_expiries(self) -> List[Union[datetime, date]]:
        """Get sorted list of expiry dates."""
        expiries = [slice_obj.expiry_date for slice_obj in self.slices.values()]
        return sorted(expiries)
    
    def get_times_to_expiry(self) -> np.ndarray:
        """Get sorted array of times to expiry."""
        times = [slice_obj.time_to_expiry for slice_obj in self.slices.values()]
        return np.array(sorted(times))
    
    def get_all_strikes(self) -> np.ndarray:
        """Get unique strikes across all expiries."""
        all_strikes = []
        for slice_obj in self.slices.values():
            all_strikes.extend(slice_obj.strikes)
        
        return np.unique(all_strikes)
    
    def get_slice_by_expiry(self, expiry: Union[datetime, date, str]) -> Optional[SmileSlice]:
        """Get smile slice for specific expiry."""
        return self.slices.get(expiry)
    
    def get_closest_expiry_slice(self, target_time: float) -> Optional[SmileSlice]:
        """Get smile slice closest to target time to expiry."""
        if not self.slices:
            return None
        
        times = [(abs(slice_obj.time_to_expiry - target_time), slice_obj) 
                 for slice_obj in self.slices.values()]
        
        _, closest_slice = min(times, key=lambda x: x[0])
        return closest_slice
    
    def interpolate_vol(self, strike: float, time_to_expiry: float) -> float:
        """
        Interpolate implied volatility at arbitrary strike and time.
        
        Uses bilinear interpolation across the vol surface.
        """
        if not self.slices:
            return np.nan
        
        # Get times and sort
        times = self.get_times_to_expiry()
        
        if time_to_expiry <= times[0]:
            # Extrapolate using first expiry
            slice_obj = self.get_closest_expiry_slice(times[0])
            return slice_obj.get_vol_at_moneyness(strike / slice_obj.forward)
        
        elif time_to_expiry >= times[-1]:
            # Extrapolate using last expiry
            slice_obj = self.get_closest_expiry_slice(times[-1])
            return slice_obj.get_vol_at_moneyness(strike / slice_obj.forward)
        
        else:
            # Interpolate between two expiries
            idx = np.searchsorted(times, time_to_expiry)
            t1, t2 = times[idx-1], times[idx]
            
            slice1 = self.get_closest_expiry_slice(t1)
            slice2 = self.get_closest_expiry_slice(t2)
            
            vol1 = slice1.get_vol_at_moneyness(strike / slice1.forward)
            vol2 = slice2.get_vol_at_moneyness(strike / slice2.forward)
            
            if np.isnan(vol1) or np.isnan(vol2):
                return np.nan
            
            # Linear interpolation in time
            weight = (time_to_expiry - t1) / (t2 - t1)
            return vol1 * (1 - weight) + vol2 * weight
    
    def get_term_structure(self, moneyness: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get term structure of implied volatility for specific moneyness.
        
        Returns:
            Tuple of (times_to_expiry, implied_vols)
        """
        times = []
        vols = []
        
        for slice_obj in self.slices.values():
            vol = slice_obj.get_vol_at_moneyness(moneyness)
            if not np.isnan(vol):
                times.append(slice_obj.time_to_expiry)
                vols.append(vol)
        
        # Sort by time
        if times:
            sort_idx = np.argsort(times)
            times = np.array(times)[sort_idx]
            vols = np.array(vols)[sort_idx]
        
        return times, vols
    
    def get_smile_params(self, time_to_expiry: float) -> Dict[str, float]:
        """
        Extract smile parameters (ATM vol, skew, convexity) for given expiry.
        """
        slice_obj = self.get_closest_expiry_slice(time_to_expiry)
        if slice_obj is None or slice_obj.implied_vols is None:
            return {}
        
        # Get ATM vol
        atm_vol = slice_obj.get_atm_vol()
        
        # Get 25-delta vols if available
        vol_25d_call = slice_obj.get_vol_at_delta(0.25, "call")
        vol_25d_put = slice_obj.get_vol_at_delta(-0.25, "put")
        
        params = {
            'atm_vol': atm_vol,
            'time_to_expiry': slice_obj.time_to_expiry
        }
        
        if not np.isnan(vol_25d_call) and not np.isnan(vol_25d_put):
            # Risk reversal and butterfly
            rr_25d = vol_25d_call - vol_25d_put
            bf_25d = 0.5 * (vol_25d_call + vol_25d_put) - atm_vol
            
            params.update({
                'vol_25d_call': vol_25d_call,
                'vol_25d_put': vol_25d_put,
                'rr_25d': rr_25d,
                'bf_25d': bf_25d
            })
        
        return params
    
    def validate_arbitrage(self) -> Dict[str, List[str]]:
        """
        Check for arbitrage violations in the volatility surface.
        
        Returns:
            Dictionary with lists of arbitrage violations by type
        """
        violations = {
            'calendar_spread': [],
            'butterfly': [],
            'call_put_parity': []
        }
        
        # Check calendar spread arbitrage
        times = self.get_times_to_expiry()
        for i in range(len(times) - 1):
            t1, t2 = times[i], times[i+1]
            slice1 = self.get_closest_expiry_slice(t1)
            slice2 = self.get_closest_expiry_slice(t2)
            
            # Calendar spreads should have positive time value
            common_strikes = np.intersect1d(slice1.strikes, slice2.strikes)
            
            for K in common_strikes:
                vol1 = slice1.get_vol_at_moneyness(K / slice1.forward)
                vol2 = slice2.get_vol_at_moneyness(K / slice2.forward)
                
                if not np.isnan(vol1) and not np.isnan(vol2):
                    # Check if total variance is increasing
                    var1 = vol1 ** 2 * t1
                    var2 = vol2 ** 2 * t2
                    
                    if var2 < var1:
                        violations['calendar_spread'].append(
                            f"Strike {K}: T1={t1:.3f} var={var1:.4f}, T2={t2:.3f} var={var2:.4f}"
                        )
        
        return violations
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert surface to pandas DataFrame."""
        rows = []
        
        for slice_obj in self.slices.values():
            for i, K in enumerate(slice_obj.strikes):
                row = {
                    'expiry_date': slice_obj.expiry_date,
                    'time_to_expiry': slice_obj.time_to_expiry,
                    'strike': K,
                    'moneyness': slice_obj.moneyness[i],
                    'log_moneyness': slice_obj.log_moneyness[i],
                    'spot': slice_obj.spot,
                    'forward': slice_obj.forward,
                    'risk_free_rate': slice_obj.risk_free_rate,
                    'dividend_yield': slice_obj.dividend_yield
                }
                
                # Add option prices if available
                if slice_obj.calls_mid is not None:
                    row['call_mid'] = slice_obj.calls_mid[i]
                if slice_obj.puts_mid is not None:
                    row['put_mid'] = slice_obj.puts_mid[i]
                if slice_obj.implied_vols is not None:
                    row['implied_vol'] = slice_obj.implied_vols[i]
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def summary_stats(self) -> Dict[str, any]:
        """Get summary statistics for the volatility surface."""
        if not self.slices:
            return {}
        
        df = self.to_dataframe()
        
        stats = {
            'n_expiries': len(self.slices),
            'n_options': len(df),
            'time_range': (df['time_to_expiry'].min(), df['time_to_expiry'].max()),
            'strike_range': (df['strike'].min(), df['strike'].max()),
            'moneyness_range': (df['moneyness'].min(), df['moneyness'].max()),
            'spot': self.spot,
            'valuation_date': self.valuation_date
        }
        
        if 'implied_vol' in df.columns:
            valid_vols = df['implied_vol'].dropna()
            if len(valid_vols) > 0:
                stats.update({
                    'vol_range': (valid_vols.min(), valid_vols.max()),
                    'avg_vol': valid_vols.mean(),
                    'vol_coverage': len(valid_vols) / len(df)
                })
        
        return stats


def build_surface_from_dataframe(
    df: pd.DataFrame,
    valuation_date: Union[datetime, date],
    spot: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0
) -> VolatilitySurface:
    """
    Build a volatility surface from option market data DataFrame.
    
    Expected DataFrame columns:
    - expiry_date: Expiry dates
    - strike: Strike prices  
    - time_to_expiry: Time to expiry in years
    - call_mid, put_mid: Option mid prices (optional)
    - call_bid, call_ask, put_bid, put_ask: Bid/ask prices (optional)
    - volume, open_interest: Liquidity metrics (optional)
    
    Args:
        df: Market data DataFrame
        valuation_date: Valuation date
        spot: Current spot price
        risk_free_rate: Risk-free rate (can be enhanced to curve later)
        dividend_yield: Dividend yield (can be enhanced to curve later)
        
    Returns:
        VolatilitySurface object
    """
    surface = VolatilitySurface(
        valuation_date=valuation_date,
        spot=spot,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield
    )
    
    # Group by expiry
    for expiry_date, group in df.groupby('expiry_date'):
        # Sort by strike
        group = group.sort_values('strike')
        
        # Get unique time to expiry for this group
        time_to_expiry = group['time_to_expiry'].iloc[0]
        
        # Calculate forward price
        forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
        
        # Extract option data
        strikes = group['strike'].values
        
        # Optional price data
        calls_mid = group['call_mid'].values if 'call_mid' in group.columns else None
        puts_mid = group['put_mid'].values if 'put_mid' in group.columns else None
        calls_bid = group['call_bid'].values if 'call_bid' in group.columns else None
        calls_ask = group['call_ask'].values if 'call_ask' in group.columns else None
        puts_bid = group['put_bid'].values if 'put_bid' in group.columns else None
        puts_ask = group['put_ask'].values if 'put_ask' in group.columns else None
        
        # Optional liquidity data
        volumes = group['volume'].values if 'volume' in group.columns else None
        open_interest = group['open_interest'].values if 'open_interest' in group.columns else None
        
        # Create smile slice
        slice_obj = SmileSlice(
            expiry_date=expiry_date,
            time_to_expiry=time_to_expiry,
            spot=spot,
            forward=forward,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            strikes=strikes,
            calls_mid=calls_mid,
            puts_mid=puts_mid,
            calls_bid=calls_bid,
            calls_ask=calls_ask,
            puts_bid=puts_bid,
            puts_ask=puts_ask,
            volumes=volumes,
            open_interest=open_interest
        )
        
        surface.add_slice(slice_obj, expiry_date)
    
    return surface