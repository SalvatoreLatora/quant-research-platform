# Quant Research Platform 🧠📈

Analisi quantitativa, ottimizzazione portafoglio e segnali ML con Streamlit.

## ✨ Funzionalità
- Logo minimal in header
- Tickers NASDAQ + S&P 500 (autogenerati via Finnhub)
- Pagina “Portafoglio” con pesi e loghi (senza salvataggi)
- AutoML (FLAML) per segnali up/down, con fallback Logistic

## 📦 Requisiti
Vedi `requirements.txt`. Python 3.10+ consigliato.

## 🔑 Secrets
Configura la tua chiave Finnhub per tickers completi e loghi:
```toml
# .streamlit/secrets.toml
FINNHUB_API_KEY = "your_finnhub_key_here"
