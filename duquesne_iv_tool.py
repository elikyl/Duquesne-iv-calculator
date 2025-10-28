import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from scipy.stats import norm
import requests
import json
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# -------------------------
# DUQUESNE COLORS
# -------------------------
# Duquesne University Official Colors
CSS_PRIMARY_COLOR = "rgb(186, 12, 47)"      # Duquesne Red
CSS_SECONDARY_COLOR = "rgb(0, 48, 135)"     # Duquesne Blue
CSS_BACKGROUND_COLOR = "rgb(240, 240, 245)"  # Light Gray
CSS_ACCENT_COLOR = "rgb(186, 12, 47)"       # Red Accent
CSS_TEXT_COLOR = "rgb(51, 51, 51)"          # Dark Gray

# For Matplotlib
PY_PRIMARY_COLOR = "#BA0C2F"      # Duquesne Red
PY_SECONDARY_COLOR = "#003087"    # Duquesne Blue
PY_BACKGROUND_COLOR = "#F0F0F5"   # Light Gray
PY_ACCENT_COLOR = "#BA0C2F"       # Red
PY_TEXT_COLOR = "#333333"         # Dark Gray

# Dark Mode Colors
DARK_BG = "#1a1a1a"
DARK_CARD = "#2d2d2d"
DARK_TEXT = "#e0e0e0"

### FOR THE FAILSAFE IF TICKER NOT FOUND
@st.cache_data
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]['Symbol'].tolist()

# -------------------------
# DARK MODE MANAGEMENT
# -------------------------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# -------------------------
# PORTFOLIO PRESETS
# -------------------------
def save_preset(name, tickers, weights):
    """Save portfolio preset to session state"""
    if 'presets' not in st.session_state:
        st.session_state.presets = {}
    st.session_state.presets[name] = {
        'tickers': tickers,
        'weights': weights
    }

def load_preset(name):
    """Load portfolio preset from session state"""
    if 'presets' in st.session_state and name in st.session_state.presets:
        return st.session_state.presets[name]
    return None

def get_preset_names():
    """Get list of saved preset names"""
    if 'presets' not in st.session_state:
        st.session_state.presets = {}
    return list(st.session_state.presets.keys())

# -------------------------
# Configuration & CSS
# -------------------------
st.set_page_config(
    page_title="Duquesne Implied Volatility & Portfolio Analysis",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dynamic CSS based on dark mode
def get_custom_css(dark_mode=False):
    if dark_mode:
        bg_color = DARK_BG
        card_bg = DARK_CARD
        text_color = DARK_TEXT
        primary = CSS_PRIMARY_COLOR
    else:
        bg_color = CSS_BACKGROUND_COLOR
        card_bg = "white"
        text_color = CSS_TEXT_COLOR
        primary = CSS_PRIMARY_COLOR
    
    return f"""
<style>
  /* Global Styles */
  .stApp {{
    background: linear-gradient(135deg, {bg_color}, {'#0a0a0a' if dark_mode else 'white'});
  }}
  
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    scroll-behavior: smooth;
  }}
  
  /* Header section */
  .header-container {{
    position: relative;
    width: 100%;
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, {CSS_PRIMARY_COLOR}, {CSS_SECONDARY_COLOR});
    border-radius: 10px;
    margin: 20px auto;
    box-shadow: 0 4px 20px rgba(186, 12, 47, 0.3);
  }}
  
  .header-title {{
    color: white;
    font-size: 48px;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    animation: fadeIn 2s ease-in-out;
  }}
  
  .header-subtitle {{
    color: white;
    font-size: 20px;
    margin-top: 10px;
    opacity: 0.9;
  }}
  
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  
  /* Top bar header */
  .top-bar {{
    background: linear-gradient(135deg, {CSS_SECONDARY_COLOR}, {CSS_PRIMARY_COLOR});
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin: 20px auto;
    width: 95%;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  }}
  
  .top-bar-content {{
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  
  .ticker-title {{
    font-size: 24px;
    font-weight: 700;
  }}
  
  .price-value {{
    font-size: 24px;
    font-weight: 600;
  }}
  
  /* Card styling */
  .card {{
    background-color: {card_bg};
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,{'0.3' if dark_mode else '0.1'});
    transition: transform 0.3s, box-shadow 0.3s;
    border-left: 4px solid {primary};
  }}
  
  .card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(186, 12, 47, 0.3);
  }}
  
  /* Section headings */
  .section-heading {{
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 15px;
    color: {primary};
    border-bottom: 2px solid {primary};
    padding-bottom: 8px;
  }}
  
  /* Metric card */
  .metric-card {{
    background-color: {'#3a3a3a' if dark_mode else '#F8F9FA'};
    border: 1px solid {'#555' if dark_mode else '#ECECEC'};
    border-radius: 8px;
    padding: 18px;
    margin-bottom: 12px;
    color: {text_color};
  }}
  
  .metric-heading {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
    color: {primary};
  }}
  
  /* Badge styling */
  .badge-iv {{
    display: inline-block;
    background-color: {CSS_PRIMARY_COLOR};
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 12px;
    margin-left: 8px;
  }}
  
  .badge-range {{
    display: inline-block;
    background-color: {CSS_SECONDARY_COLOR};
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 12px;
    margin-left: 8px;
  }}
  
  /* Button styling */
  .stButton > button {{
    background: linear-gradient(135deg, {CSS_PRIMARY_COLOR}, {CSS_SECONDARY_COLOR});
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    font-weight: 600;
    transition: all 0.3s;
    box-shadow: 0 4px 12px rgba(186, 12, 47, 0.3);
  }}
  
  .stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(186, 12, 47, 0.4);
  }}
  
  /* Download button */
  .stDownloadButton > button {{
    background: linear-gradient(135deg, {CSS_SECONDARY_COLOR}, {CSS_PRIMARY_COLOR});
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
  }}
  
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
  }}
  
  .stTabs [data-baseweb="tab"] {{
    background-color: {'#3a3a3a' if dark_mode else '#f0f0f0'};
    border-radius: 8px 8px 0 0;
    padding: 12px 24px;
    font-weight: 600;
    color: {text_color};
  }}
  
  .stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {CSS_PRIMARY_COLOR}, {CSS_SECONDARY_COLOR});
    color: white;
  }}
  
  /* Expander */
  .streamlit-expanderHeader {{
    background-color: {'#3a3a3a' if dark_mode else '#f8f9fa'};
    border-radius: 8px;
    font-weight: 600;
    color: {primary};
  }}
</style>
"""

st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Header with Duquesne branding
st.markdown("""
<div class='header-container'>
    <div class='header-title'>⚡ Duquesne Implied Volatility Tool</div>
    <div class='header-subtitle'>Advanced Financial Analysis Platform</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# GREEKS CALCULATOR
# -------------------------
class GreeksCalculator:
    """Calculate option Greeks"""
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """Calculate all Greeks for an option"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = -norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'call' else norm.cdf(-d2))) / 365
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

# -------------------------
# ENHANCED IMPLIED VOLATILITY ANALYZER
# -------------------------
class ImpliedVolatilityAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.current_price = self.stock.history(period="1d")['Close'].iloc[-1]
        self.available_expirations = self.stock.options
        self.risk_free_rate = self._get_risk_free_rate()
        
        # Get historical data for additional features
        self.hist_data = self.stock.history(period="1y")
        
    def _get_risk_free_rate(self):
        """Fetch 10-year Treasury rate"""
        try:
            treasury = yf.Ticker("^TNX")
            rate = treasury.history(period="5d")['Close'].iloc[-1] / 100
            return rate
        except:
            return 0.045

    def get_options_data(self, expiration_date):
        """Retrieve options chain for a given expiration"""
        opt = self.stock.option_chain(expiration_date)
        calls = opt.calls
        puts = opt.puts
        calls['type'] = 'call'
        puts['type'] = 'put'
        options_df = pd.concat([calls, puts], ignore_index=True)
        return options_df, expiration_date

    def _calculate_time_to_expiry(self, expiration_str):
        """Calculate time to expiry in years and datetime object"""
        exp_date = datetime.strptime(expiration_str, "%Y-%m-%d")
        today = datetime.now()
        days_to_exp = (exp_date - today).days
        years_to_exp = max(days_to_exp / 365.25, 1/365.25)
        return years_to_exp, exp_date

    def _bs_price(self, S, K, T, r, sigma, option_type):
        """Black-Scholes option pricing"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    def _bs_vega(self, S, K, T, r, sigma):
        """Vega for Newton-Raphson"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)

    def _newton_raphson_iv(self, market_price, S, K, T, r, option_type, max_iterations=100, precision=1e-8):
        """Solve for implied volatility using Newton-Raphson"""
        sigma = 0.3
        for _ in range(max_iterations):
            price = self._bs_price(S, K, T, r, sigma, option_type)
            diff = market_price - price
            if abs(diff) < precision:
                return sigma
            vega = self._bs_vega(S, K, T, r, sigma)
            if vega < 1e-10:
                break
            sigma += diff / vega
            sigma = max(0.01, min(sigma, 2.0))
        return sigma

    def calculate_iv(self, options_df, expiration):
        """Calculate IV from ATM option"""
        T, exp_dt = self._calculate_time_to_expiry(expiration)
        atm_options = options_df[
            (options_df['strike'] >= self.current_price * 0.95) &
            (options_df['strike'] <= self.current_price * 1.05) &
            (options_df['volume'] > 0)
        ].copy()
        
        if atm_options.empty:
            return np.nan, np.nan, None
        
        atm_options['mid_price'] = (atm_options['bid'] + atm_options['ask']) / 2
        atm_options = atm_options[atm_options['mid_price'] > 0.01]
        
        if atm_options.empty:
            return np.nan, np.nan, None
        
        selected = atm_options.iloc[0]
        market_price = selected['mid_price']
        strike = selected['strike']
        option_type = selected['type']
        
        iv = self._newton_raphson_iv(market_price, self.current_price, strike, T, self.risk_free_rate, option_type)
        return iv, strike, option_type

    def get_iv_percentile(self):
        """Calculate IV percentile vs historical IV"""
        try:
            # Calculate historical volatility
            returns = self.hist_data['Close'].pct_change().dropna()
            hist_vols = []
            
            # Calculate 30-day rolling volatility
            window = 30
            for i in range(window, len(returns)):
                vol = returns.iloc[i-window:i].std() * np.sqrt(252)
                hist_vols.append(vol)
            
            if not hist_vols:
                return None, None
            
            # Get current IV
            expiration_date = self.available_expirations[0]
            options_df, _ = self.get_options_data(expiration_date)
            current_iv, _, _ = self.calculate_iv(options_df, expiration_date)
            
            if np.isnan(current_iv):
                return None, None
            
            # Calculate percentile
            hist_vols = np.array(hist_vols)
            percentile = (hist_vols < current_iv).sum() / len(hist_vols) * 100
            
            return percentile, hist_vols
        except:
            return None, None

    def get_earnings_date(self):
        """Get next earnings date"""
        try:
            calendar = self.stock.calendar
            if calendar is not None and 'Earnings Date' in calendar:
                earnings_dates = calendar['Earnings Date']
                if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                    return earnings_dates[0]
                elif not isinstance(earnings_dates, list):
                    return earnings_dates
        except:
            pass
        return None

    def calculate_historical_vs_realized_vol(self):
        """Calculate historical IV vs realized volatility"""
        try:
            returns = self.hist_data['Close'].pct_change().dropna()
            
            # Calculate realized volatility (30-day rolling)
            realized_vols = []
            implied_vols = []
            dates = []
            
            window = 30
            for i in range(window, len(returns), 5):  # Sample every 5 days
                realized_vol = returns.iloc[i-window:i].std() * np.sqrt(252)
                realized_vols.append(realized_vol)
                dates.append(returns.index[i])
            
            return dates, realized_vols
        except:
            return None, None

    def get_options_chain_full(self, expiration):
        """Get full options chain for display"""
        opt = self.stock.option_chain(expiration)
        calls = opt.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        puts = opt.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        
        calls.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV']
        puts.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV']
        
        return calls, puts

    def calculate_volatility_smile(self, expiration):
        """Calculate volatility smile/skew"""
        try:
            options_df, _ = self.get_options_data(expiration)
            T, _ = self._calculate_time_to_expiry(expiration)
            
            # Filter for liquid options
            liquid_options = options_df[options_df['volume'] > 10].copy()
            
            strikes = []
            ivs = []
            
            for _, row in liquid_options.iterrows():
                strike = row['strike']
                mid_price = (row['bid'] + row['ask']) / 2
                
                if mid_price > 0.01:
                    iv = self._newton_raphson_iv(
                        mid_price, self.current_price, strike, T, 
                        self.risk_free_rate, row['type']
                    )
                    if not np.isnan(iv) and iv > 0:
                        strikes.append(strike)
                        ivs.append(iv * 100)
            
            return strikes, ivs
        except:
            return None, None

    def display_iv_metrics(self):
        """Enhanced IV metrics display"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>📊 Implied Volatility Metrics</h2>", unsafe_allow_html=True)
        
        ivs = []
        dates_list = []
        for exp_date in self.available_expirations[:4]:
            options_df, expiration = self.get_options_data(exp_date)
            iv, strike, flag = self.calculate_iv(options_df, expiration)
            ivs.append(iv)
            dates_list.append(exp_date)
        
        # Display current metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if not np.isnan(ivs[0]):
            with col1:
                st.metric("Current IV", f"{ivs[0]*100:.2f}%", help="Implied Volatility from nearest expiration")
            
            # IV Percentile
            percentile, hist_vols = self.get_iv_percentile()
            with col2:
                if percentile is not None:
                    st.metric("IV Percentile", f"{percentile:.1f}%", help="Current IV vs 1-year history")
                else:
                    st.metric("IV Percentile", "N/A")
            
            # Earnings date
            with col3:
                earnings_date = self.get_earnings_date()
                if earnings_date:
                    st.metric("Next Earnings", earnings_date.strftime("%Y-%m-%d") if hasattr(earnings_date, 'strftime') else str(earnings_date))
                else:
                    st.metric("Next Earnings", "Unknown")
            
            # Current price
            with col4:
                st.metric("Current Price", f"${self.current_price:.2f}")
        
        # Expected movements
        st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>Expected Price Movements</h3>", unsafe_allow_html=True)
        
        for iv, exp_date in zip(ivs, dates_list):
            if not np.isnan(iv):
                t, exp_dt = self._calculate_time_to_expiry(exp_date)
                move = self.current_price * iv * np.sqrt(t)
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<strong>{exp_date}</strong>: ±${move:.2f} (±{iv*np.sqrt(t)*100:.2f}%)"
                    f"</div>",
                    unsafe_allow_html=True
                )

    def display_greeks(self):
        """Display option Greeks"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>🎯 Option Greeks</h2>", unsafe_allow_html=True)
        
        try:
            expiration_date = self.available_expirations[0]
            options_df, expiration = self.get_options_data(expiration_date)
            iv, strike, flag = self.calculate_iv(options_df, expiration)
            
            if not np.isnan(iv):
                T, _ = self._calculate_time_to_expiry(expiration_date)
                greeks = GreeksCalculator.calculate_greeks(
                    self.current_price, strike, T, self.risk_free_rate, iv, flag
                )
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Delta", f"{greeks['delta']:.4f}", help="Rate of change in option price per $1 change in stock")
                with col2:
                    st.metric("Gamma", f"{greeks['gamma']:.4f}", help="Rate of change in delta per $1 change in stock")
                with col3:
                    st.metric("Theta", f"{greeks['theta']:.4f}", help="Time decay per day")
                with col4:
                    st.metric("Vega", f"{greeks['vega']:.4f}", help="Sensitivity to 1% change in IV")
                with col5:
                    st.metric("Rho", f"{greeks['rho']:.4f}", help="Sensitivity to 1% change in interest rates")
                
                # Greeks explanation
                st.info(f"""
                **Greeks for ATM {flag.upper()} option** (Strike: ${strike:.2f}, Expiry: {expiration_date})
                
                - **Delta ({greeks['delta']:.4f})**: Option price changes by ${abs(greeks['delta']):.2f} for every $1 move in stock
                - **Gamma ({greeks['gamma']:.4f})**: Delta changes by {greeks['gamma']:.4f} for every $1 move in stock
                - **Theta ({greeks['theta']:.4f})**: Option loses ${abs(greeks['theta']):.2f} per day due to time decay
                - **Vega ({greeks['vega']:.4f})**: Option price changes by ${greeks['vega']:.2f} for 1% change in IV
                """)
        except Exception as e:
            st.warning(f"Could not calculate Greeks: {e}")

    def display_historical_iv_chart(self):
        """Display historical IV timeline"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>📈 Historical Volatility Analysis</h2>", unsafe_allow_html=True)
        
        dates, realized_vols = self.calculate_historical_vs_realized_vol()
        
        if dates and realized_vols:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=[v * 100 for v in realized_vols],
                mode='lines',
                name='Realized Volatility (30-day)',
                line=dict(color=PY_PRIMARY_COLOR, width=2)
            ))
            
            fig.update_layout(
                title="Historical Realized Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate historical volatility chart")

    def display_volatility_smile(self):
        """Display volatility smile/skew"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>😊 Volatility Smile / Skew</h2>", unsafe_allow_html=True)
        
        expiration = self.available_expirations[0]
        strikes, ivs = self.calculate_volatility_smile(expiration)
        
        if strikes and ivs:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=strikes,
                y=ivs,
                mode='markers+lines',
                name='Implied Volatility',
                marker=dict(size=8, color=PY_PRIMARY_COLOR),
                line=dict(color=PY_SECONDARY_COLOR, width=2)
            ))
            
            # Add current price line
            fig.add_vline(
                x=self.current_price,
                line_dash="dash",
                line_color="green",
                annotation_text="Current Price"
            )
            
            fig.update_layout(
                title=f"Volatility Smile/Skew ({expiration})",
                xaxis_title="Strike Price ($)",
                yaxis_title="Implied Volatility (%)",
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate volatility smile chart")

    def display_options_chain(self):
        """Display full options chain"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>📋 Options Chain</h2>", unsafe_allow_html=True)
        
        expiration = st.selectbox("Select Expiration:", self.available_expirations)
        
        calls, puts = self.get_options_chain_full(expiration)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<h3 style='color: {CSS_SECONDARY_COLOR};'>CALLS</h3>", unsafe_allow_html=True)
            st.dataframe(calls, use_container_width=True, height=400)
        
        with col2:
            st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>PUTS</h3>", unsafe_allow_html=True)
            st.dataframe(puts, use_container_width=True, height=400)

    def monte_carlo_simulation(self, num_simulations=1000, time_horizon_days=30):
        """Monte Carlo price simulation"""
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Simulating {num_simulations} potential price paths over {time_horizon_days} days using implied volatility...</p>", unsafe_allow_html=True)
        
        expiration_date = self.available_expirations[0]
        options_df, expiration = self.get_options_data(expiration_date)
        iv, strike, flag = self.calculate_iv(options_df, expiration)
        
        if np.isnan(iv):
            st.error("Could not calculate IV for simulation")
            return
        
        dt = 1 / 252
        num_steps = time_horizon_days
        
        paths = np.zeros((num_steps, num_simulations))
        paths[0] = self.current_price
        
        for t in range(1, num_steps):
            rand = np.random.standard_normal(num_simulations)
            paths[t] = paths[t-1] * np.exp((self.risk_free_rate - 0.5 * iv**2) * dt + iv * np.sqrt(dt) * rand)
        
        # Create interactive plot with plotly
        fig = go.Figure()
        
        # Plot sample paths
        for i in range(min(100, num_simulations)):
            fig.add_trace(go.Scatter(
                y=paths[:, i],
                mode='lines',
                line=dict(width=0.5, color='lightblue'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add mean path
        mean_path = paths.mean(axis=1)
        fig.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(width=3, color=PY_PRIMARY_COLOR)
        ))
        
        # Add confidence intervals
        upper_bound = np.percentile(paths, 95, axis=1)
        lower_bound = np.percentile(paths, 5, axis=1)
        
        fig.add_trace(go.Scatter(
            y=upper_bound,
            mode='lines',
            name='95th Percentile',
            line=dict(width=2, dash='dash', color=PY_SECONDARY_COLOR)
        ))
        
        fig.add_trace(go.Scatter(
            y=lower_bound,
            mode='lines',
            name='5th Percentile',
            line=dict(width=2, dash='dash', color=PY_SECONDARY_COLOR)
        ))
        
        fig.update_layout(
            title=f"Monte Carlo Simulation - {num_simulations} Paths",
            xaxis_title="Days",
            yaxis_title="Price ($)",
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        final_prices = paths[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Final Price", f"${final_prices.mean():.2f}")
        with col2:
            st.metric("Median Final Price", f"${np.median(final_prices):.2f}")
        with col3:
            st.metric("5th Percentile", f"${np.percentile(final_prices, 5):.2f}")
        with col4:
            st.metric("95th Percentile", f"${np.percentile(final_prices, 95):.2f}")

    def display_data_for_excel(self):
        """Generate Excel export data"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>📥 Data Export for Excel</h2>", unsafe_allow_html=True)
        
        ivs = []
        dates_list = []
        for exp_date in self.available_expirations[:4]:
            options_df, expiration = self.get_options_data(exp_date)
            iv, strike, flag = self.calculate_iv(options_df, expiration)
            ivs.append(iv)
            dates_list.append(exp_date)
        
        # Create downloadable CSV
        export_data = []
        for iv, exp_date in zip(ivs, dates_list):
            if not np.isnan(iv):
                t, exp_dt = self._calculate_time_to_expiry(exp_date)
                move_dollars = self.current_price * iv * np.sqrt(t)
                move_percent = iv * np.sqrt(t) * 100
                export_data.append({
                    'Expiration Date': exp_date,
                    'Implied Volatility (%)': iv * 100,
                    'Expected Move ($)': move_dollars,
                    'Expected Move (%)': move_percent
                })
        
        df = pd.DataFrame(export_data)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{self.ticker}_iv_data.csv",
            mime="text/csv"
        )
        
        st.dataframe(df, use_container_width=True)
        
        # Text format for copy-paste
        st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>Copy-Paste Format</h3>", unsafe_allow_html=True)
        iv_chart = "Expiration Date\tImplied Volatility (%)\tExpected Move ($)\tExpected Move (%)\n"
        for iv, exp_date in zip(ivs, dates_list):
            if not np.isnan(iv):
                t, _ = self._calculate_time_to_expiry(exp_date)
                move_dollars = self.current_price * iv * np.sqrt(t)
                move_percent = iv * np.sqrt(t) * 100
                iv_chart += f"{exp_date}\t{iv*100:.2f}\t{move_dollars:.2f}\t{move_percent:.2f}\n"
        
        st.code(iv_chart.strip(), language="text")

    def generate_pdf_report(self):
        """Generate PDF report"""
        try:
            buffer = BytesIO()
            
            with PdfPages(buffer) as pdf:
                # Page 1: IV Metrics
                fig, ax = plt.subplots(figsize=(11, 8.5))
                fig.patch.set_facecolor('white')
                ax.axis('off')
                
                # Title
                ax.text(0.5, 0.95, f'{self.ticker} Implied Volatility Report', 
                       ha='center', va='top', fontsize=20, fontweight='bold',
                       color=PY_PRIMARY_COLOR)
                
                ax.text(0.5, 0.90, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                       ha='center', va='top', fontsize=10, color='gray')
                
                # Get data
                ivs = []
                dates_list = []
                for exp_date in self.available_expirations[:4]:
                    options_df, expiration = self.get_options_data(exp_date)
                    iv, strike, flag = self.calculate_iv(options_df, expiration)
                    ivs.append(iv)
                    dates_list.append(exp_date)
                
                # Current metrics
                y_pos = 0.80
                ax.text(0.1, y_pos, 'Current Metrics:', fontsize=14, fontweight='bold',
                       color=PY_PRIMARY_COLOR)
                y_pos -= 0.05
                
                if not np.isnan(ivs[0]):
                    ax.text(0.1, y_pos, f'Current Price: ${self.current_price:.2f}', fontsize=11)
                    y_pos -= 0.04
                    ax.text(0.1, y_pos, f'Current IV: {ivs[0]*100:.2f}%', fontsize=11)
                    y_pos -= 0.06
                    
                    ax.text(0.1, y_pos, 'Expected Movements:', fontsize=14, fontweight='bold',
                           color=PY_PRIMARY_COLOR)
                    y_pos -= 0.05
                    
                    for iv, exp_date in zip(ivs, dates_list):
                        if not np.isnan(iv):
                            t, _ = self._calculate_time_to_expiry(exp_date)
                            move = self.current_price * iv * np.sqrt(t)
                            ax.text(0.1, y_pos, f'{exp_date}: ±${move:.2f} (±{iv*np.sqrt(t)*100:.2f}%)',
                                   fontsize=10)
                            y_pos -= 0.04
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Page 2: Historical volatility chart
                dates, realized_vols = self.calculate_historical_vs_realized_vol()
                if dates and realized_vols:
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.plot(dates, [v * 100 for v in realized_vols], 
                           color=PY_PRIMARY_COLOR, linewidth=2)
                    ax.set_title('Historical Realized Volatility', 
                                fontsize=16, fontweight='bold', color=PY_PRIMARY_COLOR)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Volatility (%)', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            buffer.seek(0)
            return buffer
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            return None

# -------------------------
# ENHANCED PORTFOLIO ANALYZER
# -------------------------
class PortfolioImpliedVolatilityAnalyzer:
    def __init__(self, tickers, weights, total_portfolio_value):
        self.tickers = [t for t in tickers if t]
        self.weights = np.array([w/100 for w in weights if w is not None])[:len(self.tickers)]
        self.total_portfolio_value = total_portfolio_value
        self.stocks = {ticker: yf.Ticker(ticker) for ticker in self.tickers}
        self.risk_free_rate = self._get_risk_free_rate()
        
        # Get historical data for all stocks
        self.hist_data = {}
        for ticker in self.tickers:
            try:
                self.hist_data[ticker] = self.stocks[ticker].history(period="1y")
            except:
                pass

    def _get_risk_free_rate(self):
        try:
            treasury = yf.Ticker("^TNX")
            rate = treasury.history(period="5d")['Close'].iloc[-1] / 100
            return rate
        except:
            return 0.045

    def calculate_correlation_matrix(self):
        """Calculate correlation matrix of returns"""
        returns_dict = {}
        
        for ticker in self.tickers:
            if ticker in self.hist_data:
                returns = self.hist_data[ticker]['Close'].pct_change().dropna()
                returns_dict[ticker] = returns
        
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            corr_matrix = returns_df.corr()
            return corr_matrix
        return None

    def display_correlation_heatmap(self):
        """Display correlation heatmap"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>🔥 Correlation Heatmap</h2>", unsafe_allow_html=True)
        
        corr_matrix = self.calculate_correlation_matrix()
        
        if corr_matrix is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       ax=ax, vmin=-1, vmax=1)
            ax.set_title('Portfolio Correlation Matrix', fontsize=14, fontweight='bold',
                        color=PY_PRIMARY_COLOR)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Could not generate correlation matrix")

    def display_risk_return_scatter(self):
        """Display risk vs return scatter plot"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>📊 Risk vs Return Analysis</h2>", unsafe_allow_html=True)
        
        risk_return_data = []
        
        for i, ticker in enumerate(self.tickers):
            if ticker in self.hist_data:
                returns = self.hist_data[ticker]['Close'].pct_change().dropna()
                annual_return = returns.mean() * 252 * 100
                annual_risk = returns.std() * np.sqrt(252) * 100
                
                risk_return_data.append({
                    'Ticker': ticker,
                    'Return (%)': annual_return,
                    'Risk (%)': annual_risk,
                    'Weight (%)': self.weights[i] * 100
                })
        
        if risk_return_data:
            df = pd.DataFrame(risk_return_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['Risk (%)'],
                y=df['Return (%)'],
                mode='markers+text',
                marker=dict(
                    size=df['Weight (%)'] * 2,
                    color=df['Weight (%)'],
                    colorscale=[[0, PY_SECONDARY_COLOR], [1, PY_PRIMARY_COLOR]],
                    showscale=True,
                    colorbar=dict(title="Weight (%)")
                ),
                text=df['Ticker'],
                textposition='top center',
                name='Holdings'
            ))
            
            fig.update_layout(
                title='Portfolio Holdings: Risk vs Return',
                xaxis_title='Annual Risk (%)',
                yaxis_title='Annual Return (%)',
                template='plotly_white',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Could not generate risk/return analysis")

    def display_portfolio_iv(self):
        """Enhanced portfolio IV display"""
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>📊 Portfolio Implied Volatility Analysis</h2>", unsafe_allow_html=True)
        
        # Calculate individual IVs
        individual_ivs = []
        for ticker in self.tickers:
            try:
                analyzer = ImpliedVolatilityAnalyzer(ticker)
                expiration_date = analyzer.available_expirations[0]
                options_df, expiration = analyzer.get_options_data(expiration_date)
                iv, _, _ = analyzer.calculate_iv(options_df, expiration)
                individual_ivs.append(iv if not np.isnan(iv) else 0)
            except:
                individual_ivs.append(0)
        
        individual_ivs = np.array(individual_ivs)
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix()
        
        if corr_matrix is not None:
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            portfolio_iv = np.sqrt(
                np.dot(self.weights**2, individual_ivs**2) +
                avg_correlation * np.sum([self.weights[i] * self.weights[j] * individual_ivs[i] * individual_ivs[j]
                                         for i in range(len(self.weights))
                                         for j in range(i+1, len(self.weights))])
            )
        else:
            portfolio_iv = np.sqrt(np.dot(self.weights**2, individual_ivs**2))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio IV", f"{portfolio_iv*100:.2f}%")
        with col2:
            st.metric("Avg Correlation", f"{avg_correlation:.2f}" if corr_matrix is not None else "N/A")
        with col3:
            st.metric("Portfolio Value", f"${self.total_portfolio_value:,.2f}")
        
        # Individual stock IVs
        st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>Individual Stock IVs</h3>", unsafe_allow_html=True)
        
        for i, (ticker, iv, weight) in enumerate(zip(self.tickers, individual_ivs, self.weights)):
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>{ticker}</strong> ({weight*100:.1f}%): IV = {iv*100:.2f}%"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Expected movements
        st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>Expected Portfolio Movements</h3>", unsafe_allow_html=True)
        
        time_frames = [30, 60, 90]
        for days in time_frames:
            t = days / 365.25
            move_dollars = self.total_portfolio_value * portfolio_iv * np.sqrt(t)
            move_percent = portfolio_iv * np.sqrt(t) * 100
            st.markdown(
                f"<div class='metric-card'>"
                f"<strong>{days} Days</strong>: ±${move_dollars:,.2f} (±{move_percent:.2f}%)"
                f"</div>",
                unsafe_allow_html=True
            )

# -------------------------
# TICKER COMPARISON
# -------------------------
def display_ticker_comparison(tickers_to_compare):
    """Compare multiple tickers side by side"""
    st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>🔍 Multi-Ticker Comparison</h2>", unsafe_allow_html=True)
    
    comparison_data = []
    
    for ticker in tickers_to_compare:
        if ticker:
            try:
                analyzer = ImpliedVolatilityAnalyzer(ticker)
                expiration_date = analyzer.available_expirations[0]
                options_df, expiration = analyzer.get_options_data(expiration_date)
                iv, strike, _ = analyzer.calculate_iv(options_df, expiration)
                
                percentile, _ = analyzer.get_iv_percentile()
                
                comparison_data.append({
                    'Ticker': ticker,
                    'Price': f"${analyzer.current_price:.2f}",
                    'IV (%)': f"{iv*100:.2f}" if not np.isnan(iv) else "N/A",
                    'IV Percentile': f"{percentile:.1f}%" if percentile else "N/A",
                    'ATM Strike': f"${strike:.2f}" if not np.isnan(strike) else "N/A"
                })
            except Exception as e:
                st.warning(f"Could not analyze {ticker}: {e}")
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Visual comparison
        try:
            analyzers = []
            labels = []
            ivs = []
            
            for ticker in tickers_to_compare:
                if ticker:
                    try:
                        analyzer = ImpliedVolatilityAnalyzer(ticker)
                        expiration_date = analyzer.available_expirations[0]
                        options_df, expiration = analyzer.get_options_data(expiration_date)
                        iv, _, _ = analyzer.calculate_iv(options_df, expiration)
                        if not np.isnan(iv):
                            labels.append(ticker)
                            ivs.append(iv * 100)
                    except:
                        pass
            
            if labels and ivs:
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=ivs,
                        marker_color=[PY_PRIMARY_COLOR, PY_SECONDARY_COLOR, PY_ACCENT_COLOR][:len(labels)]
                    )
                ])
                
                fig.update_layout(
                    title='Implied Volatility Comparison',
                    xaxis_title='Ticker',
                    yaxis_title='IV (%)',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except:
            pass

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
st.sidebar.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>⚙️ Settings</h2>", unsafe_allow_html=True)

# Dark mode toggle
dark_mode_btn = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
if dark_mode_btn != st.session_state.dark_mode:
    toggle_dark_mode()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>📍 Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.selectbox(
    "Select Page:",
    ["Single Stock Analysis", "Portfolio Analysis", "Ticker Comparison"]
)

# -------------------------
# PAGE 1: SINGLE STOCK ANALYSIS
# -------------------------
if page == "Single Stock Analysis":
    st.markdown(f"<h1 style='color: {CSS_PRIMARY_COLOR};'>📈 Single Stock IV Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Comprehensive implied volatility analysis for individual stocks</p>", unsafe_allow_html=True)
    
    ticker_input = st.text_input("Enter Stock Ticker:", "AAPL", key="single_stock_ticker")
    
    if st.button("🚀 Analyze", key="analyze_single"):
        try:
            with st.spinner("Analyzing..."):
                analyzer = ImpliedVolatilityAnalyzer(ticker_input)
                
                tabs = st.tabs([
                    "📊 Overview",
                    "🎯 Greeks",
                    "😊 Vol Smile",
                    "📈 Historical",
                    "🎲 Monte Carlo",
                    "📋 Options Chain",
                    "📥 Export"
                ])
                
                with tabs[0]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_iv_metrics()
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[1]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_greeks()
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[2]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_volatility_smile()
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[3]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_historical_iv_chart()
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[4]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>🎲 Monte Carlo Simulation</h2>", unsafe_allow_html=True)
                    analyzer.monte_carlo_simulation()
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[5]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_options_chain()
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[6]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_data_for_excel()
                    
                    # PDF Export
                    st.markdown("---")
                    st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>📄 PDF Report</h3>", unsafe_allow_html=True)
                    
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating PDF..."):
                            pdf_buffer = analyzer.generate_pdf_report()
                            if pdf_buffer:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"{ticker_input}_iv_report.pdf",
                                    mime="application/pdf"
                                )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error: {e}")

# -------------------------
# PAGE 2: PORTFOLIO ANALYSIS
# -------------------------
elif page == "Portfolio Analysis":
    st.markdown(f"<h1 style='color: {CSS_PRIMARY_COLOR};'>📊 Portfolio IV Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Analyze implied volatility across your entire portfolio</p>", unsafe_allow_html=True)
    
    # Portfolio presets
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        preset_name = st.text_input("Preset Name:", placeholder="My Portfolio")
    with col2:
        if st.button("💾 Save Preset"):
            if preset_name:
                # Get current values (we'll populate these from session state if available)
                save_preset(preset_name, 
                          st.session_state.get('portfolio_tickers', []),
                          st.session_state.get('portfolio_weights', []))
                st.success(f"Saved preset: {preset_name}")
    with col3:
        preset_names = get_preset_names()
        if preset_names:
            selected_preset = st.selectbox("Load Preset:", [""] + preset_names)
            if selected_preset:
                preset = load_preset(selected_preset)
                if preset:
                    st.session_state.portfolio_tickers = preset['tickers']
                    st.session_state.portfolio_weights = preset['weights']
                    st.rerun()
    
    st.markdown("---")
    
    total_portfolio_value = st.number_input(
        "Total Portfolio Value ($):",
        min_value=0.0,
        value=1000000.0,
        step=10000.0,
        format="%.2f"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Stock Tickers</h4>", unsafe_allow_html=True)
        tickers = []
        for i in range(12):
            default_val = st.session_state.get('portfolio_tickers', [''] * 12)[i] if i < len(st.session_state.get('portfolio_tickers', [])) else ''
            val = st.text_input(f"Stock {i+1}:", value=default_val, key=f"ticker_{i}")
            tickers.append(val.upper() if val else "")
    
    with col2:
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Weights (%)</h4>", unsafe_allow_html=True)
        weights = []
        for i in range(12):
            default_weight = st.session_state.get('portfolio_weights', [0.0] * 12)[i] if i < len(st.session_state.get('portfolio_weights', [])) else 0.0
            weight = st.number_input(
                f"Weight {i+1} (%):",
                min_value=0.0,
                max_value=100.0,
                value=default_weight,
                step=0.1,
                key=f"weight_{i}"
            )
            weights.append(weight)
    
    # Save to session state
    st.session_state.portfolio_tickers = tickers
    st.session_state.portfolio_weights = weights
    
    if st.button("📊 Calculate Portfolio IV", key="calc_portfolio"):
        total_weight = sum(w for w in weights if w is not None)
        
        if abs(total_weight - 100.0) > 0.01:
            st.error(f"❌ Total weight must equal 100%. Current total: {total_weight:.2f}%")
        elif total_portfolio_value <= 0:
            st.error("❌ Total portfolio value must be greater than zero")
        else:
            try:
                with st.spinner("Analyzing portfolio..."):
                    port_analyzer = PortfolioImpliedVolatilityAnalyzer(tickers, weights, total_portfolio_value)
                    
                    tabs = st.tabs(["📊 Portfolio IV", "🔥 Correlations", "📈 Risk/Return"])
                    
                    with tabs[0]:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        port_analyzer.display_portfolio_iv()
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with tabs[1]:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        port_analyzer.display_correlation_heatmap()
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with tabs[2]:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        port_analyzer.display_risk_return_scatter()
                        st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# -------------------------
# PAGE 3: TICKER COMPARISON
# -------------------------
elif page == "Ticker Comparison":
    st.markdown(f"<h1 style='color: {CSS_PRIMARY_COLOR};'>🔍 Multi-Ticker Comparison</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Compare implied volatility metrics across multiple stocks</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ticker1 = st.text_input("Ticker 1:", "AAPL", key="compare_1")
    with col2:
        ticker2 = st.text_input("Ticker 2:", "MSFT", key="compare_2")
    with col3:
        ticker3 = st.text_input("Ticker 3:", "GOOGL", key="compare_3")
    with col4:
        ticker4 = st.text_input("Ticker 4:", "", key="compare_4")
    
    tickers_to_compare = [t.upper() for t in [ticker1, ticker2, ticker3, ticker4] if t]
    
    if st.button("🔍 Compare", key="compare_btn"):
        if len(tickers_to_compare) < 2:
            st.error("❌ Please enter at least 2 tickers to compare")
        else:
            try:
                with st.spinner("Comparing tickers..."):
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    display_ticker_comparison(tickers_to_compare)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {CSS_TEXT_COLOR}; padding: 20px;'>
    <p><strong>Duquesne Implied Volatility Analysis Tool</strong></p>
    <p>Built with Streamlit • Enhanced Financial Analytics Platform</p>
    <p style='font-size: 12px; color: gray;'>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
