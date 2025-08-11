import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import requests
from pathlib import Path
from datetime import datetime, timedelta

# ML / AutoML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.covariance import LedoitWolf

try:
    from flaml import AutoML
except Exception:
    AutoML = None

st.set_page_config(page_title="Quant Research Platform", page_icon="📈", layout="wide")

# ---------------- UI: Header con logo ----------------
st.markdown("""
<style>
.header-bar{display:flex;align-items:center;gap:12px;margin-bottom:8px;}
.header-title{font-size:1.1rem;font-weight:700;color:#E6F1FF;margin:0;padding:0;}
</style>
""", unsafe_allow_html=True)

def load_svg(path:str)->str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return '''
        <svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
          <rect width="28" height="28" rx="6" fill="#0B0F13" stroke="#1f2a36"/>
          <polyline points="4,20 8,14 12,16 17,8 23,10" fill="none" stroke="#00E5FF" stroke-width="2" stroke-linecap="round"/>
        </svg>
        '''

logo_svg = load_svg("assets/qrp_logo.svg")
st.markdown(f"""
<div class="header-bar">
  <div>{logo_svg}</div>
  <div class="header-title">Quant Research Platform</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Tickers: autogenera CSV (NASDAQ + S&P 500) con Finnhub ----------------
def ensure_ticker_csv(csv_path="tickers_sp500_nasdaq.csv", finnhub_key:str=""):
    p = Path(csv_path)
    if p.exists() and p.stat().st_size > 10_000:
        return p
    if not finnhub_key:
        return p  # senza chiave, lasceremo un fallback in memoria
    try:
        r1 = requests.get(
            "https://finnhub.io/api/v1/index/constituents",
            params={"symbol": "^GSPC", "token": finnhub_key},
            timeout=15
        )
        sp = r1.json().get("constituents", [])

        r2 = requests.get(
            "https://finnhub.io/api/v1/stock/symbol",
            params={"exchange": "US", "token": finnhub_key},
            timeout=30
        )
        us = r2.json()

        rows, seen = [], set()
        for x in us:
            if str(x.get("mic", "")).upper() == "XNAS" and x.get("symbol"):
                sym = x["symbol"]
                if sym not in seen:
                    rows.append({"symbol": sym, "name": x.get("description", "")})
                    seen.add(sym)
        for sym in sp:
            if sym not in seen:
                rows.append({"symbol": sym, "name": ""})
                seen.add(sym)

        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(p, index=False)
    except Exception:
        pass
    return p

FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
csv_path = ensure_ticker_csv(finnhub_key=FINNHUB_API_KEY)

def load_tickers(csv_path: Path):
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            syms = df["symbol"].dropna().astype(str).unique().tolist()
            names = df["name"].fillna("").astype(str).tolist()
            return syms, df
        except Exception:
            pass
    # Fallback essenziale
    fallback = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","NFLX","INTC",
        "JPM","V","UNH","XOM","PG","PFE","KO","PEP","CSCO","AVGO"
    ]
    return fallback, pd.DataFrame({"symbol": fallback, "name": [""]*len(fallback)})

ALL_TICKERS, TICKERS_DF = load_tickers(csv_path)

# ---------------- Funzioni utili ----------------
def fetch_prices(tickers, start, end, interval="1d"):
    if not tickers:
        return pd.DataFrame()
    df = yf.download(
        tickers=tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False
    )["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all").ffill().dropna()
    df.columns = [c.split(" ")[0] for c in df.columns]  # semplifica colonne multi-ticker
    return df

def inverse_vol_weights(returns: pd.DataFrame):
    vol = returns.std().replace(0, np.nan).dropna()
    w = 1.0 / vol
    w = w / w.sum()
    return w

def min_variance_weights(returns: pd.DataFrame, non_negative=True):
    X = returns.dropna()
    if X.shape[1] == 0:
        return pd.Series(dtype=float)
    cov = LedoitWolf().fit(X.values).covariance_
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + 1e-8 * np.eye(n))
    ones = np.ones(n)
    raw = inv @ ones
    w = raw / raw.sum()
    if non_negative:
        w = np.clip(w, 0, None)
        s = w.sum()
        w = w / s if s > 0 else np.ones_like(w)/len(w)
    return pd.Series(w, index=X.columns)

def compute_portfolio_weights(prices: pd.DataFrame, method: str):
    rets = prices.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    if method == "Equal Weight":
        w = pd.Series(1.0 / rets.shape[1], index=rets.columns)
    elif method == "Inverse Volatility":
        w = inverse_vol_weights(rets)
    else:  # Min Variance (Ledoit-Wolf)
        w = min_variance_weights(rets, non_negative=True)
    return w

def build_features(close: pd.Series):
    df = pd.DataFrame({"close": close})
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["mom_5"] = df["close"].pct_change(5)
    # RSI(14)
    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).rolling(14).mean()
    roll_down = pd.Series(down, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()
    X = df[["ret1","ret5","ma5","ma20","mom_5","rsi14"]]
    y = df["target"]
    return df, X, y

def automl_binary_classification(X_train, y_train, X_eval, time_budget=20):
    if AutoML is None:
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_eval)[:, 1]
        return model, proba
    automl = AutoML()
    automl.fit(
        X_train=X_train, y_train=y_train,
        task="classification",
        time_budget=time_budget,
        metric="log_loss",
        eval_method="holdout",
        estimator_list=["lgbm","xgboost","rf","extra_tree","lrl1","catboost"],
        verbose=0
    )
    try:
        proba = automl.predict_proba(X_eval)[:, 1]
    except Exception:
        pred = automl.predict(X_eval)
        proba = (pred - np.min(pred)) / (np.ptp(pred) + 1e-9)
    return automl, proba

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Parametri")
    default_start = datetime.utcnow().date() - timedelta(days=365)
    start = st.date_input("Data inizio", value=default_start)
    end = st.date_input("Data fine", value=datetime.utcnow().date())
    interval = st.selectbox("Intervallo prezzi", ["1d","1h","1wk"], index=0)

    method = st.selectbox("Metodo pesi portafoglio", ["Equal Weight","Inverse Volatility","Min Variance"])

    st.markdown("### Selezione titoli")
    selected = st.multiselect(
        "Scegli i ticker",
        options=ALL_TICKERS,
        default=["AAPL","MSFT","NVDA","AMZN","GOOGL"]
        if set(["AAPL","MSFT","NVDA","AMZN","GOOGL"]).issubset(set(ALL_TICKERS)) else ALL_TICKERS[:5]
    )

    st.markdown("---")
    st.markdown("### Modello ML (AutoML)")
    ml_ticker = st.selectbox("Ticker per ML", options=selected if selected else ALL_TICKERS, index=0)
    p_long = st.slider("Soglia LONG (prob.)", 0.50, 0.70, 0.55, step=0.01)
    p_short = st.slider("Soglia SHORT (prob.)", 0.30, 0.50, 0.45, step=0.01)
    time_budget = st.slider("Tempo AutoML (s)", 5, 60, 20, step=5)

    btn_calc = st.button("Calcola portafoglio")

# ---------------- Workflow principale ----------------
st.markdown("## 📊 Analisi e portafoglio")

if btn_calc and selected:
    prices = fetch_prices(selected, start, end, interval)
    if prices.empty:
        st.warning("Dati prezzi non disponibili per i parametri selezionati.")
    else:
        weights = compute_portfolio_weights(prices, method)
        weights_df = (
            weights.rename("Peso")
            .reset_index()
            .rename(columns={"index": "Ticker"})
        )
        st.success("Portafoglio calcolato.")
        # Salva in sessione per la pagina Portafoglio
        st.session_state["portfolio_weights_df"] = weights_df.copy()
        st.session_state["portfolio_updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
        st.session_state["portfolio_method"] = method
        st.session_state["provider_key"] = FINNHUB_API_KEY  # per loghi su 02_Portafoglio
        # Mostra anteprima pesi
        st.subheader("Pesi del portafoglio")
        st.dataframe(weights_df.style.format({"Peso": "{:.2%}"}), use_container_width=True)

        # Grafico prezzi aggregati
        st.subheader("Prezzi storici")
        st.line_chart(prices)

# ---------------- Sezione ML: AutoML per segnali ----------------
st.markdown("---")
st.markdown("## 🤖 Segnali ML (AutoML)")

if ml_ticker:
    try:
        s = yf.download(ml_ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)["Close"]
        s = s.dropna()
        df_feat, X, y = build_features(s)
        if len(df_feat) < 200:
            st.info("Serie troppo corta per addestrare un modello robusto (min ~200 campioni).")
        else:
            n = len(X)
            split = int(n * 0.8)
            X_train, y_train = X.iloc[:split], y.iloc[:split]
            X_test, y_test = X.iloc[split:], y.iloc[split:]
            model, proba = automl_binary_classification(X_train, y_train, X_test, time_budget=time_budget)
            y_pred = (proba > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Ticker: {ml_ticker} — Accuracy holdout: {acc:.3f}")
            # Genera segnali su tutta la serie
            _, _, y_all = build_features(s)
            _, proba_all = automl_binary_classification(X_train, y_train, X, time_budget=min(5, time_budget))
            sig = np.where(proba_all > p_long, 1, np.where(proba_all < p_short, -1, 0))
            out = df_feat.iloc[:len(sig)][["close"]].copy()
            out["proba_up"] = proba_all[:len(out)]
            out["signal"] = sig[:len(out)]
            st.line_chart(out[["close"]].rename(columns={"close": "Prezzo"}))
            st.area_chart(out[["proba_up"]].rename(columns={"proba_up": "Prob. Up"}))
            st.dataframe(out.tail(20), use_container_width=True)
    except Exception as e:
        st.warning(f"Impossibile addestrare il modello per {ml_ticker}.")
else:
    st.info("Seleziona almeno un ticker nella sidebar per la sezione ML.")

# ---------------- Note UI ----------------
st.caption("Suggerimento: aggiungi una chiave Finnhub nei Secrets per autogenerare i tickers NASDAQ + S&P 500 e mostrare i loghi.")
