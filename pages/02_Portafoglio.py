import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

st.set_page_config(page_title="Quant Research Platform — Portafoglio", page_icon="🧩", layout="wide")

st.markdown("""
<style>
.section-card{background:#121821;border:1px solid #1f2a36;border-radius:10px;padding:14px 16px;margin-bottom:10px;}
.cell-logo{display:flex;align-items:center;gap:10px}
th, td { border-bottom: 1px solid #1f2a36; padding: 8px 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧩 Composizione del portafoglio")

wdf = st.session_state.get("portfolio_weights_df")
updated_at = st.session_state.get("portfolio_updated_at")
method = st.session_state.get("portfolio_method", "Equal Weight")
provider_key = st.session_state.get("provider_key", "")
provider_name = "Finnhub" if provider_key else ""

if wdf is None or wdf.empty:
    st.info("Nessun portafoglio trovato. Torna alla pagina principale, seleziona i ticker e calcola i pesi.")
    st.stop()

st.markdown(
    f'<div class="section-card">Metodo di ottimizzazione: <b>{method}</b> — ultimo aggiornamento: {updated_at or "n/d"}</div>',
    unsafe_allow_html=True
)

def fetch_profile_finnhub(symbol: str, key: str):
    if not key:
        return {}
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": symbol, "token": key},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

rows = []
for _, row in wdf.iterrows():
    sym = row["Ticker"]
    peso = float(row["Peso"]) * 100.0
    name = ""
    logo = ""
    prof = fetch_profile_finnhub(sym, provider_key) if provider_name == "Finnhub" else {}
    name = prof.get("name", "")
    logo = prof.get("logo", "")
    rows.append({"Ticker": sym, "Azienda": name, "LogoURL": logo, "Peso%": round(peso, 2)})

table_df = pd.DataFrame(rows).sort_values("Peso%", ascending=False)

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    html_rows = []
    for _, r in table_df.iterrows():
        left = (
            f'<div class="cell-logo"><img src="{r["LogoURL"]}" width="22"/> '
            f'<b>{r["Ticker"]}</b> — {r["Azienda"]}</div>'
            if r["LogoURL"]
            else f'<div class="cell-logo"><b>{r["Ticker"]}</b> — {r["Azienda"]}</div>'
        )
        html_rows.append(
            f"<tr><td>{left}</td><td style='text-align:right;'>{r['Peso%']:.2f}%</td></tr>"
        )
    html = """
    <table style="width:100%; border-collapse:collapse;">
      <thead><tr><th style="text-align:left;">Titolo</th><th style="text-align:right;">Peso</th></tr></thead>
      <tbody>
    """ + "\n".join(html_rows) + "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption("I loghi sono caricati da URL pubblico, senza salvataggi su disco. Se non disponibili gratuitamente, non vengono mostrati.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if not table_df.empty:
        fig = px.pie(table_df, names="Ticker", values="Peso%", hole=0.45, title="Pesi del portafoglio")
        fig.update_layout(height=380, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
