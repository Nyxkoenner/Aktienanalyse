import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from typing import List, Dict, Optional
import feedparser
from dateutil import parser as dateparser
import altair as alt
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time


# -----------------------------
# HTTP / Caching Helpers
# -----------------------------

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def fetch_html(url: str, retries: int = 3, backoff: float = 1.2, timeout: int = 20) -> str:
    """
    Holt HTML via requests mit User-Agent + Retry/Backoff.
    """
    last_err = None
    for i in range(retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"HTTP Fehler bei {url}: {last_err}")


def read_html_tables(html: str) -> List[pd.DataFrame]:
    """
    Liest alle HTML-Tabellen aus einem HTML-String.
    """
    # pd.read_html braucht meistens lxml
    return pd.read_html(html)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    """
    Findet eine passende Spalte anhand von Kandidaten (case-insensitive).
    """
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fuzzy: enthält?
    for c in columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def disk_cache_load(name: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    path = CACHE_DIR / f"{name}.csv"
    if not path.exists():
        return None
    age_sec = time.time() - path.stat().st_mtime
    if age_sec > max_age_hours * 3600:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def disk_cache_save(name: str, df: pd.DataFrame) -> None:
    path = CACHE_DIR / f"{name}.csv"
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass


# -----------------------------
# Index-Konstituenten laden
# -----------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_index_constituents(index_name: str) -> pd.DataFrame:
    """
    Lädt Index-Mitglieder als DataFrame mit Spalten:
      [name, ticker_yahoo, sector]

    - DAX 40: versucht en.wikipedia (meist stabilere Tabelle mit Ticker)
             fallback: de.wikipedia (falls verfügbar)
    - S&P 500: en.wikipedia (List_of_S%26P_500_companies)
    """

    # Erst Disk-Cache versuchen (entlastet Wikipedia massiv)
    cache_key = f"constituents_{index_name.replace(' ', '_').lower()}"
    cached = disk_cache_load(cache_key, max_age_hours=24)
    if cached is not None and not cached.empty:
        return cached

    try:
        if index_name == "DAX 40":
            # Primär: englische Seite (liefert typischerweise Ticker symbol)
            urls = [
                "https://en.wikipedia.org/wiki/DAX",
                "https://de.wikipedia.org/wiki/DAX",
            ]

            df_final = None

            for url in urls:
                html = fetch_html(url)
                tables = read_html_tables(html)

                # passende Tabelle suchen: Name + Ticker
                for t in tables:
                    t = normalize_columns(t)
                    cols = list(t.columns)

                    name_col = find_col(cols, ["Company", "Unternehmen", "Name", "Constituent", "Security"])
                    ticker_col = find_col(cols, ["Ticker symbol", "Ticker", "Symbol", "WKN", "ISIN"])
                    sector_col = find_col(cols, ["Industry", "Sector", "Industrie", "Branche"])

                    # Wir brauchen für yfinance zwingend einen Ticker (Ticker/Symbol).
                    # WKN/ISIN hilft nicht direkt. Daher akzeptieren wir nur echte Ticker/Symbol-Spalten.
                    if ticker_col is None:
                        continue

                    # Heuristik: ticker_col muss “ticker”/“symbol” im Namen haben (nicht nur ISIN/WKN)
                    tcl = ticker_col.lower()
                    if ("isin" in tcl) or ("wkn" in tcl):
                        continue

                    if name_col is None:
                        # notfalls irgendeine “Company”-ähnliche Spalte
                        name_col = find_col(cols, ["Company", "Unternehmen", "Security", "Name"])
                    if name_col is None:
                        continue

                    if sector_col is None:
                        sector_col = "sector_tmp"
                        t[sector_col] = None

                    df_tmp = t[[name_col, ticker_col, sector_col]].copy()
                    df_tmp.columns = ["name", "ticker_yahoo", "sector"]

                    # Ticker normalisieren (DAX -> .DE)
                    df_tmp["ticker_yahoo"] = (
                        df_tmp["ticker_yahoo"].astype(str).str.strip()
                    )
                    df_tmp = df_tmp[df_tmp["ticker_yahoo"].str.len() > 0]

                    # Manche Tabellen enthalten Fußnoten / Sonderzeichen
                    df_tmp["ticker_yahoo"] = df_tmp["ticker_yahoo"].str.replace(r"\[.*?\]", "", regex=True).str.strip()

                    # .DE anfügen, wenn nicht schon ein Suffix vorhanden
                    df_tmp["ticker_yahoo"] = df_tmp["ticker_yahoo"].apply(
                        lambda x: x if "." in x else f"{x}.DE"
                    )

                    # Duplikate entfernen
                    df_tmp = df_tmp.drop_duplicates(subset=["ticker_yahoo"]).reset_index(drop=True)

                    # Plausibilitätscheck: wir erwarten ~40 Werte
                    if len(df_tmp) >= 30:
                        df_final = df_tmp
                        break

                if df_final is not None:
                    break

            if df_final is None or df_final.empty:
                st.error(
                    "Keine passende DAX-Tabelle mit Ticker/Symbol gefunden. "
                    "Wikipedia liefert manchmal nur ISIN/WKN in der Zusammensetzung."
                )
                return pd.DataFrame()

            disk_cache_save(cache_key, df_final)
            return df_final[["name", "ticker_yahoo", "sector"]]

        elif index_name == "S&P 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            html = fetch_html(url)
            tables = read_html_tables(html)

            df_final = None
            for t in tables:
                t = normalize_columns(t)
                cols = list(t.columns)

                name_col = find_col(cols, ["Security", "Company", "Name"])
                ticker_col = find_col(cols, ["Symbol", "Ticker", "Ticker symbol"])
                sector_col = find_col(cols, ["GICS Sector", "Sector"])

                if name_col and ticker_col and sector_col:
                    df_tmp = t[[name_col, ticker_col, sector_col]].copy()
                    df_tmp.columns = ["name", "ticker_yahoo", "sector"]

                    df_tmp["ticker_yahoo"] = df_tmp["ticker_yahoo"].astype(str).str.strip()

                    # yfinance: BRK.B -> BRK-B, BF.B -> BF-B, etc.
                    df_tmp["ticker_yahoo"] = df_tmp["ticker_yahoo"].str.replace(".", "-", regex=False)

                    df_tmp = df_tmp.drop_duplicates(subset=["ticker_yahoo"]).reset_index(drop=True)

                    if len(df_tmp) >= 400:
                        df_final = df_tmp
                        break

            if df_final is None or df_final.empty:
                st.error("Keine passende S&P-500-Tabelle gefunden.")
                return pd.DataFrame()

            disk_cache_save(cache_key, df_final)
            return df_final[["name", "ticker_yahoo", "sector"]]

        else:
            st.error("Unbekannter Index")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Konnte {index_name} nicht laden: {e}")
        return pd.DataFrame()


# -----------------------------
# Prozent-Helper
# -----------------------------

def to_percent(val) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if abs(v) <= 1:
        return v * 100.0
    return v


# -----------------------------
# Kennzahlen & Kursdaten laden
# -----------------------------

def infer_dividend_frequency(div_series: pd.Series) -> Optional[str]:
    if div_series is None or div_series.empty:
        return None
    last_date = div_series.index.max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    recent = div_series[div_series.index >= one_year_ago]
    n = len(recent)
    if n == 0:
        return None
    if n >= 4:
        return "quarterly"
    if 2 <= n <= 3:
        return "semiannual"
    if n == 1:
        return "annual"
    return "irregular"


def calc_dividend_growth_5y(div_series: pd.Series) -> Optional[float]:
    if div_series is None or div_series.empty:
        return None
    yearly = div_series.groupby(div_series.index.year).sum().sort_index()
    if len(yearly) < 2:
        return None
    first_year = yearly.index[0]
    last_year = yearly.index[-1]
    n_years = last_year - first_year
    if n_years <= 0:
        return None
    if n_years > 5:
        first_year = last_year - 5
        yearly = yearly[yearly.index >= first_year]
        if len(yearly) < 2:
            return None
    first_val = float(yearly.iloc[0])
    last_val = float(yearly.iloc[-1])
    if first_val <= 0 or last_val <= 0:
        return None
    n_years = yearly.index[-1] - yearly.index[0]
    if n_years <= 0:
        return None
    cagr = (last_val / first_val) ** (1 / n_years) - 1
    return cagr * 100.0


def fetch_metrics(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {}
    errors: List[str] = []

    for ticker in tickers:
        metrics: Dict[str, Optional[float]] = {
            "last_price": None,
            "change_1d": None,
            "change_5d": None,
            "change_1y": None,
            "vol_30d": None,
            "vol_1y": None,
            "market_cap": None,
            "pe_ratio": None,
            "forward_pe": None,
            "pb_ratio": None,
            "ps_ratio": None,
            "ev_ebitda": None,
            "net_margin": None,
            "operating_margin": None,
            "roe": None,
            "roa": None,
            "dividend_yield": None,
            "dividend_per_share": None,
            "payout_ratio": None,
            "dividend_growth_5y": None,
            "dividend_frequency": None,
            "debt_to_equity": None,
            "net_debt_ebitda": None,
            "history": None,
        }

        try:
            t = yf.Ticker(ticker)

            hist = t.history(period="1y")
            if hist.empty:
                errors.append(f"{ticker}: keine Kursdaten erhalten.")
            else:
                close = hist["Close"]

                if ticker.endswith(".L") and close.iloc[-1] > 100:
                    close = close / 100.0

                metrics["history"] = close
                last_price = float(close.iloc[-1])
                metrics["last_price"] = last_price

                rets = close.pct_change().dropna()

                def pct_change_from(idx_offset: int) -> Optional[float]:
                    if len(close) <= idx_offset:
                        return None
                    base = float(close.iloc[-1 - idx_offset])
                    if base == 0:
                        return None
                    return (last_price - base) / base * 100.0

                metrics["change_1d"] = pct_change_from(1)
                metrics["change_5d"] = pct_change_from(5)
                metrics["change_1y"] = pct_change_from(len(close) - 1)

                if len(rets) >= 30:
                    metrics["vol_30d"] = float(rets.tail(30).std() * sqrt(252) * 100.0)
                if len(rets) > 0:
                    metrics["vol_1y"] = float(rets.std() * sqrt(252) * 100.0)

            try:
                info = t.info

                metrics["market_cap"] = info.get("marketCap")
                metrics["pe_ratio"] = info.get("trailingPE")
                metrics["forward_pe"] = info.get("forwardPE")
                metrics["pb_ratio"] = info.get("priceToBook")
                metrics["ps_ratio"] = info.get("priceToSalesTrailing12Months")
                metrics["ev_ebitda"] = info.get("enterpriseToEbitda")

                metrics["net_margin"] = to_percent(info.get("profitMargins"))
                metrics["operating_margin"] = to_percent(info.get("operatingMargins"))
                metrics["roe"] = to_percent(info.get("returnOnEquity"))
                metrics["roa"] = to_percent(info.get("returnOnAssets"))

                dy_raw = info.get("dividendYield")
                metrics["dividend_yield"] = to_percent(dy_raw)
                metrics["dividend_per_share"] = info.get("dividendRate")
                metrics["payout_ratio"] = to_percent(info.get("payoutRatio"))

                # Div-Yield Fallback
                try:
                    lp = metrics.get("last_price")
                    dps = metrics.get("dividend_per_share")
                    if lp not in (None, 0) and dps not in (None, 0):
                        est_yield = float(dps) / float(lp) * 100.0
                        dy_current = metrics["dividend_yield"]
                        if dy_current is None or dy_current > 25 or abs(dy_current - est_yield) > 15:
                            metrics["dividend_yield"] = est_yield
                except Exception:
                    pass

                dte_raw = info.get("debtToEquity")
                if dte_raw is not None:
                    dte = float(dte_raw)
                    if dte > 0:
                        metrics["debt_to_equity"] = dte / 100.0

                total_debt = info.get("totalDebt")
                total_cash = info.get("totalCash")
                ebitda = info.get("ebitda")
                if total_debt is not None and total_cash is not None and ebitda not in (None, 0):
                    net_debt = float(total_debt) - float(total_cash)
                    metrics["net_debt_ebitda"] = float(net_debt) / float(ebitda)

            except Exception as e_info:
                errors.append(f"{ticker}: Fehler beim Laden von info – {e_info}")

            try:
                divs = t.dividends
                if divs is not None and not divs.empty:
                    metrics["dividend_frequency"] = infer_dividend_frequency(divs)
                    metrics["dividend_growth_5y"] = calc_dividend_growth_5y(divs)
            except Exception as e_div:
                errors.append(f"{ticker}: Fehler bei Dividendenhistorie – {e_div}")

        except Exception as e:
            errors.append(f"{ticker}: Allgemeiner Fehler – {e}")

        result[ticker] = metrics

    if errors:
        st.sidebar.error("Probleme beim Laden der Daten:")
        for msg in errors[:30]:
            st.sidebar.write("- ", msg)
        if len(errors) > 30:
            st.sidebar.write(f"... und {len(errors) - 30} weitere")

    return result


# -----------------------------
# News & Sentiment
# -----------------------------

POSITIVE_WORDS = {
    "schlägt", "rekord", "stark", "upgrades", "kaufen", "outperform",
    "bullish", "profit", "gewinn", "wachstum", "besser als erwartet", "angehoben"
}
NEGATIVE_WORDS = {
    "warnung", "gewinnwarnung", "verfehlt", "herabgestuft", "verkaufen",
    "schwach", "verlust", "betrug", "untersuchung", "klage", "senken"
}

RSS_SOURCES: List[str] = [
    "https://www.tagesschau.de/xml/rss2",
    "https://www.tagesschau.de/wirtschaft/unternehmen/index~rss2.xml",
    "https://www.finanzen.net/rss/news",
    "https://www.onvista.de/news/feed/aktien",
    "https://www.handelsblatt.com/contentexport/feed/wirtschaft",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.eqs-news.com/de/news/dgap/rss",
]

def simple_sentiment(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0
    for w in POSITIVE_WORDS:
        if w in t:
            score += 1
    for w in NEGATIVE_WORDS:
        if w in t:
            score -= 1
    return float(score)

def fetch_news_for_ticker(
    ticker: str,
    company_name: str = "",
    days_back: int = 30,
    max_items: int = 50,
) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    entries = []

    keywords = set()
    if ticker:
        keywords.add(ticker.lower())

    if company_name:
        name = company_name.lower()
        for suf in [" ag", " se", " sa", " s.a.", " plc", " n.v.", " gmbh",
                    " & co. kg", " kgaa", " kg"]:
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        name = name.strip()
        if name:
            keywords.add(name)
            for part in name.split():
                if len(part) > 2:
                    keywords.add(part)

    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        if not getattr(feed, "entries", None):
            continue

        for entry in feed.entries:
            title = entry.get("title", "") or ""
            summary = entry.get("summary", "") or ""
            link = entry.get("link", "") or ""

            published_raw = entry.get("published", None)
            published = None

            if published_raw:
                try:
                    published = dateparser.parse(published_raw)
                except Exception:
                    published = None

            if published is None and hasattr(entry, "published_parsed"):
                try:
                    ts = entry.published_parsed
                    published = datetime(
                        ts.tm_year, ts.tm_mon, ts.tm_mday,
                        ts.tm_hour, ts.tm_min, ts.tm_sec,
                        tzinfo=timezone.utc,
                    )
                except Exception:
                    published = None

            if published is None:
                continue

            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            else:
                published = published.astimezone(timezone.utc)

            if published < cutoff:
                continue

            text_full = f"{title} {summary}".lower()
            if keywords and not any(k in text_full for k in keywords):
                continue

            sent_score = simple_sentiment(text_full)
            sent_label = "positiv" if sent_score > 0 else "negativ" if sent_score < 0 else "neutral"

            entries.append({
                "published": published,
                "title": title,
                "link": link,
                "sentiment_score": sent_score,
                "sentiment_label": sent_label,
            })

    if not entries:
        return pd.DataFrame(columns=["published", "title", "link", "sentiment_score", "sentiment_label"])

    return (
        pd.DataFrame(entries)
        .sort_values("published", ascending=False)
        .head(max_items)
        .reset_index(drop=True)
    )


# -----------------------------
# Graph
# -----------------------------

def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        ticker = row["ticker_yahoo"]
        sector = row.get("sector", None)
        price = row.get("last_price", None)

        label_price = "n/a" if pd.isna(price) else f"{price:.2f}"
        label = f"{ticker}\n{label_price}"

        G.add_node(ticker, label=label, sector=sector)

    for sector in df["sector"].dropna().unique():
        same_sector = df[df["sector"] == sector]["ticker_yahoo"].tolist()
        for i in range(len(same_sector)):
            for j in range(i + 1, len(same_sector)):
                G.add_edge(same_sector[i], same_sector[j], relation="same_sector")

    return G

def draw_graph(G: nx.Graph):
    if len(G.nodes) == 0:
        return None

    pos = nx.circular_layout(G)  # ohne scipy

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()}, font_size=8, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# -----------------------------
# Score & Styling
# -----------------------------

def compute_total_score(row: pd.Series) -> Optional[float]:
    score = 0.0
    max_score = 0.0

    roe = row.get("roe")
    if pd.notna(roe):
        max_score += 25
        score += 25 if roe >= 20 else 20 if roe >= 15 else 15 if roe >= 10 else 10 if roe >= 5 else 0

    net_margin = row.get("net_margin")
    if pd.notna(net_margin):
        max_score += 20
        score += 20 if net_margin >= 20 else 16 if net_margin >= 15 else 12 if net_margin >= 10 else 8 if net_margin >= 5 else 0

    div_yield = row.get("dividend_yield")
    if pd.notna(div_yield):
        max_score += 20
        score += 20 if 2 <= div_yield <= 6 else 10 if 1 <= div_yield < 2 else 8 if 6 < div_yield <= 10 else 0

    dte = row.get("debt_to_equity")
    if pd.notna(dte):
        max_score += 15
        score += 15 if dte < 0.5 else 12 if dte < 1.0 else 8 if dte < 2.0 else 4 if dte < 3.0 else 0

    vol_1y = row.get("vol_1y")
    if pd.notna(vol_1y):
        max_score += 20
        score += 20 if vol_1y <= 20 else 15 if vol_1y <= 30 else 8 if vol_1y <= 40 else 0

    if max_score == 0:
        return None
    return round(score / max_score * 100.0, 1)

def colorize_change(val):
    if pd.isna(val):
        return ""
    color = "green" if val > 0 else "red" if val < 0 else "black"
    return f"color: {color}"

def colorize_score(val):
    if pd.isna(val):
        return ""
    color = "green" if val >= 70 else "orange" if val >= 40 else "red"
    return f"color: {color}"

SMA_COLORS = {
    "SMA20": "#ff4d4d",
    "SMA50": "#ff9dbb",
    "SMA200": "#2ec4b6",
}


# -----------------------------
# Portfolio
# -----------------------------

def build_portfolio_view(df_with_metrics: pd.DataFrame, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if df_with_metrics is None or df_with_metrics.empty:
        return pd.DataFrame()
    if portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame()

    required_cols = {"ticker_yahoo", "shares", "cost_basis"}
    if not required_cols.issubset(set(portfolio_df.columns)):
        missing = required_cols - set(portfolio_df.columns)
        st.error(f"Im Portfolio fehlen Spalten: {missing}")
        return pd.DataFrame()

    df_metrics = df_with_metrics.copy()
    port = portfolio_df.copy()

    df_metrics["ticker_yahoo"] = df_metrics["ticker_yahoo"].astype(str).str.strip()
    port["ticker_yahoo"] = port["ticker_yahoo"].astype(str).str.strip()

    known_tickers = set(df_metrics["ticker_yahoo"].unique())
    portfolio_tickers = set(port["ticker_yahoo"].unique())
    missing_tickers = sorted(portfolio_tickers - known_tickers)

    if missing_tickers:
        st.info(f"Zusätzliche Ticker werden für das Portfolio nachgeladen: {', '.join(missing_tickers)}")
        extra_metrics_dict = fetch_metrics(missing_tickers)

        extra_rows = []
        for t in missing_tickers:
            m = extra_metrics_dict.get(t, {})
            row = {"ticker_yahoo": t}
            for k, v in m.items():
                if k == "history":
                    continue
                row[k] = v
            row.setdefault("name", t)
            row.setdefault("sector", None)
            extra_rows.append(row)

        if extra_rows:
            df_extra = pd.DataFrame(extra_rows)
            df_metrics = pd.concat([df_metrics, df_extra], ignore_index=True)

        if "total_score" not in df_metrics.columns:
            df_metrics["total_score"] = df_metrics.apply(compute_total_score, axis=1)
        else:
            mask = df_metrics["total_score"].isna()
            if mask.any():
                df_metrics.loc[mask, "total_score"] = df_metrics.loc[mask].apply(compute_total_score, axis=1)

    merged = port.merge(df_metrics, on="ticker_yahoo", how="left", suffixes=("", "_yahoo"))

    merged["shares"] = merged["shares"].astype(float)
    merged["cost_basis"] = merged["cost_basis"].astype(float)

    merged["market_value"] = merged["shares"] * merged["last_price"]
    merged["cost_value"] = merged["shares"] * merged["cost_basis"]

    merged["pnl_abs"] = merged["market_value"] - merged["cost_value"]
    merged["pnl_pct"] = merged["pnl_abs"] / merged["cost_value"] * 100.0

    total_mv = merged["market_value"].sum(skipna=True)
    merged["weight_pct"] = (merged["market_value"] / total_mv * 100.0) if total_mv > 0 else 0.0

    merged["dividend_income_est"] = (
        merged["market_value"] * merged["dividend_yield"].fillna(0) / 100.0
        if "dividend_yield" in merged.columns else 0.0
    )

    return merged


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.title("Aktien Explorer – Mini-Tool mit Fundamentaldaten")

    index_name = st.sidebar.selectbox("Index auswählen", ["DAX 40", "S&P 500"])

    df = load_index_constituents(index_name)
    if df.empty:
        st.stop()

    # --- Filter ---
    st.sidebar.header("Filter")
    sector_options = ["Alle"] + sorted(df["sector"].dropna().unique().tolist())
    selected_sector = st.sidebar.selectbox("Sektor", sector_options)
    search_text = st.sidebar.text_input("Suche nach Firmenname oder Ticker (optional)").strip()

    # --- Daten laden ---
    st.subheader("Schritt 1: Kurse, Performance & Fundamentaldaten laden")

    max_stocks = st.sidebar.slider(
        "Max. Anzahl Unternehmen für Analyse",
        min_value=20,
        max_value=min(500, len(df)),
        value=min(80, len(df)),
        step=20,
        help="Je mehr Unternehmen, desto länger dauert der Datenabruf über yfinance."
    )

    # index-spezifische Session Keys (damit DAX und S&P sich nicht überschreiben)
    key_df = f"df_with_metrics__{index_name}"
    key_metrics = f"metrics__{index_name}"

    if st.button("Daten laden / aktualisieren", key=f"btn_reload_{index_name}"):
        tickers = df["ticker_yahoo"].tolist()[:max_stocks]
        df = df[df["ticker_yahoo"].isin(tickers)].reset_index(drop=True)

        with st.spinner("Lade Daten über yfinance ..."):
            metrics = fetch_metrics(tickers)

        def map_metric(col_name: str):
            return df["ticker_yahoo"].map(lambda t: metrics.get(t, {}).get(col_name))

        # Kurs & Performance
        df["last_price"] = map_metric("last_price")
        df["change_1d"] = map_metric("change_1d")
        df["change_5d"] = map_metric("change_5d")
        df["change_1y"] = map_metric("change_1y")
        df["vol_30d"] = map_metric("vol_30d")
        df["vol_1y"] = map_metric("vol_1y")

        # Bewertung & Profitabilität
        df["market_cap"] = map_metric("market_cap")
        df["pe_ratio"] = map_metric("pe_ratio")
        df["forward_pe"] = map_metric("forward_pe")
        df["pb_ratio"] = map_metric("pb_ratio")
        df["ps_ratio"] = map_metric("ps_ratio")
        df["ev_ebitda"] = map_metric("ev_ebitda")
        df["net_margin"] = map_metric("net_margin")
        df["operating_margin"] = map_metric("operating_margin")
        df["roe"] = map_metric("roe")
        df["roa"] = map_metric("roa")

        # Dividende & Verschuldung
        df["dividend_yield"] = map_metric("dividend_yield")
        df["dividend_per_share"] = map_metric("dividend_per_share")
        df["payout_ratio"] = map_metric("payout_ratio")
        df["dividend_growth_5y"] = map_metric("dividend_growth_5y")
        df["dividend_frequency"] = map_metric("dividend_frequency")
        df["debt_to_equity"] = map_metric("debt_to_equity")
        df["net_debt_ebitda"] = map_metric("net_debt_ebitda")

        df["total_score"] = df.apply(compute_total_score, axis=1)

        st.session_state[key_metrics] = metrics
        st.session_state[key_df] = df.copy()
        st.success("Daten aktualisiert!")

    else:
        if key_df in st.session_state:
            df = st.session_state[key_df]
        else:
            # Spalten auffüllen, damit UI nicht crasht
            for col in [
                "last_price", "change_1d", "change_5d", "change_1y",
                "vol_30d", "vol_1y", "market_cap", "pe_ratio", "forward_pe",
                "pb_ratio", "ps_ratio", "ev_ebitda", "net_margin",
                "operating_margin", "roe", "roa", "dividend_yield",
                "dividend_per_share", "payout_ratio", "dividend_growth_5y",
                "dividend_frequency", "debt_to_equity", "net_debt_ebitda",
                "total_score",
            ]:
                if col not in df.columns:
                    df[col] = None

    # --- Filter anwenden ---
    df_view = df.copy()
    if selected_sector != "Alle":
        df_view = df_view[df_view["sector"] == selected_sector]

    if search_text:
        mask = (
            df_view["name"].astype(str).str.contains(search_text, case=False, na=False)
            | df_view["ticker_yahoo"].astype(str).str.contains(search_text, case=False, na=False)
        )
        df_view = df_view[mask]

    # --- Tabs ---
    tab_overview, tab_funda, tab_risk, tab_sector, tab_news, tab_portfolio = st.tabs(
        ["Überblick", "Fundamentaldaten", "Risiko-Panel", "Sektor-Übersicht", "News & Events", "Portfolio"]
    )

    with tab_overview:
        st.subheader("Überblick: Kurs & Performance")
        overview_cols = [
            "name", "ticker_yahoo", "sector", "last_price",
            "change_1d", "change_5d", "change_1y",
            "vol_30d", "vol_1y", "total_score",
        ]
        styled_overview = (
            df_view[overview_cols]
            .style.format(
                {
                    "last_price": "{:.2f}",
                    "change_1d": "{:+.2f}%",
                    "change_5d": "{:+.2f}%",
                    "change_1y": "{:+.2f}%",
                    "vol_30d": "{:.2f}%",
                    "vol_1y": "{:.2f}%",
                    "total_score": "{:.1f}",
                },
                na_rep="–",
            )
            .applymap(colorize_change, subset=["change_1d", "change_5d", "change_1y"])
            .applymap(colorize_score, subset=["total_score"])
        )
        st.dataframe(styled_overview, use_container_width=True)

    with tab_funda:
        st.subheader("Fundamentaldaten")
        df_funda = df_view.copy()
        df_funda["market_cap_billion"] = df_funda["market_cap"] / 1e9

        funda_cols = [
            "name", "ticker_yahoo", "total_score", "market_cap_billion",
            "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda",
            "net_margin", "operating_margin", "roe", "roa",
            "dividend_yield", "dividend_per_share", "payout_ratio",
            "dividend_growth_5y", "dividend_frequency",
            "debt_to_equity", "net_debt_ebitda",
        ]

        styled_funda = (
            df_funda[funda_cols]
            .style.format(
                {
                    "market_cap_billion": "{:.1f}",
                    "pe_ratio": "{:.2f}",
                    "forward_pe": "{:.2f}",
                    "pb_ratio": "{:.2f}",
                    "ps_ratio": "{:.2f}",
                    "ev_ebitda": "{:.2f}",
                    "net_margin": "{:.2f}%",
                    "operating_margin": "{:.2f}%",
                    "roe": "{:.2f}%",
                    "roa": "{:.2f}%",
                    "dividend_yield": "{:.2f}%",
                    "dividend_per_share": "{:.2f}",
                    "payout_ratio": "{:.2f}%",
                    "dividend_growth_5y": "{:.2f}%",
                    "debt_to_equity": "{:.2f}x",
                    "net_debt_ebitda": "{:.2f}",
                    "total_score": "{:.1f}",
                },
                na_rep="–",
            )
            .applymap(colorize_change, subset=["net_margin", "operating_margin", "roe", "roa", "dividend_yield", "dividend_growth_5y"])
            .applymap(colorize_score, subset=["total_score"])
        )
        st.dataframe(styled_funda, use_container_width=True)

    with tab_risk:
        st.subheader("Risiko-Panel für einzelne Aktie")
        if df_view.empty:
            st.info("Keine Firmen für diese Filtereinstellung.")
        else:
            ticker_choice = st.selectbox("Ticker auswählen", options=df_view["ticker_yahoo"].tolist())
            row = df_view[df_view["ticker_yahoo"] == ticker_choice].iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Volatilität 1J", f"{row['vol_1y']:.1f}%" if pd.notna(row["vol_1y"]) else "–")
            col2.metric("Debt/Equity (x)", f"{row['debt_to_equity']:.2f}x" if pd.notna(row["debt_to_equity"]) else "–")
            col3.metric("Gesamt-Score", f"{row['total_score']:.1f}" if pd.notna(row["total_score"]) else "–")

    with tab_sector:
        st.subheader("Sektor-Übersicht")
        if df_view.empty:
            st.info("Keine Firmen für diese Filtereinstellung.")
        else:
            sector_stats = (
                df_view.groupby("sector")
                .agg(
                    anzahl=("ticker_yahoo", "count"),
                    avg_change_1y=("change_1y", "mean"),
                    avg_score=("total_score", "mean"),
                )
                .sort_values("avg_change_1y", ascending=False)
            )
            st.dataframe(sector_stats.style.format({"avg_change_1y": "{:+.2f}%", "avg_score": "{:.1f}"}, na_rep="–"), use_container_width=True)
            st.bar_chart(sector_stats["avg_change_1y"])

    with tab_news:
        st.subheader("News & Events")
        if df_view.empty:
            st.info("Keine Firmen für diese Filtereinstellung.")
        else:
            news_ticker = st.selectbox("Ticker für News auswählen", options=df_view["ticker_yahoo"].tolist(), key="news_ticker_select")
            days_back = st.slider("Zeitraum (Tage)", min_value=3, max_value=90, value=30, step=1)

            if st.button("News laden", key="load_news_button"):
                row_news = df_view[df_view["ticker_yahoo"] == news_ticker].iloc[0]
                company_name = row_news["name"]

                news_df = fetch_news_for_ticker(news_ticker, company_name=company_name, days_back=days_back, max_items=50)
                st.session_state["news_df"] = news_df

            news_df = st.session_state.get("news_df")

            if news_df is None or news_df.empty:
                st.info("Noch keine News geladen oder keine Einträge im Feed.")
            else:
                st.dataframe(news_df[["published", "title", "sentiment_label", "link"]], use_container_width=True)

    with tab_portfolio:
        st.subheader("Dein Portfolio")
        if key_df not in st.session_state:
            st.info("Bitte zuerst oben auf **'Daten laden / aktualisieren'** klicken.")
        else:
            df_metrics_full = st.session_state[key_df]

            try:
                portfolio_df = pd.read_csv("portfolio.csv")
                st.success("Portfolio aus 'portfolio.csv' geladen.")
            except FileNotFoundError:
                st.error("❌ Datei 'portfolio.csv' nicht im Projektordner gefunden.")
                portfolio_df = None
            except Exception as e_port:
                st.error(f"❌ Fehler beim Laden der Portfolio-Datei: {e_port}")
                portfolio_df = None

            if portfolio_df is not None and not portfolio_df.empty:
                st.dataframe(portfolio_df, use_container_width=True)
                port_view = build_portfolio_view(df_metrics_full, portfolio_df)
                st.dataframe(port_view, use_container_width=True)

    # --- Mini-Chart / Sparkline ---
    st.subheader("Schritt 3: Mini-Kurscharts (Sparkline)")

    metrics_store = st.session_state.get(key_metrics, {})
    if metrics_store:
        available_tickers = df_view["ticker_yahoo"].tolist()
        if available_tickers:
            selected_ticker = st.selectbox("Wähle einen Ticker für den Mini-Chart:", options=available_tickers)
            period_choice = st.selectbox("Zeitraum", options=["2 Monate", "6 Monate", "1 Jahr", "5 Jahre"], index=0)
            show_smas = st.checkbox("SMA 20/50/200 anzeigen", value=True)

            base_series = metrics_store.get(selected_ticker, {}).get("history", None)

            series = base_series
            if period_choice == "5 Jahre":
                try:
                    t = yf.Ticker(selected_ticker)
                    hist5 = t.history(period="5y")
                    if not hist5.empty:
                        series = hist5["Close"]
                except Exception:
                    pass

            if series is not None and not series.empty:
                if period_choice == "2 Monate":
                    series_window = series.tail(min(60, len(series)))
                elif period_choice == "6 Monate":
                    series_window = series.tail(min(130, len(series)))
                elif period_choice == "1 Jahr":
                    series_window = series.tail(min(252, len(series)))
                else:
                    series_window = series

                spark = series_window.to_frame(name="Kurs")
                spark.index.name = "Datum"

                if show_smas:
                    spark["SMA20"] = spark["Kurs"].rolling(window=20).mean()
                    spark["SMA50"] = spark["Kurs"].rolling(window=50).mean()
                    spark["SMA200"] = spark["Kurs"].rolling(window=200).mean()

                chart_df = spark.reset_index()
                chart_df = chart_df.rename(columns={chart_df.columns[0]: "Datum"})

                base_chart = alt.Chart(chart_df).mark_line(strokeWidth=1.8).encode(
                    x=alt.X("Datum:T", title="Datum"),
                    y=alt.Y("Kurs:Q", title="Preis"),
                    tooltip=["Datum", "Kurs"],
                )

                sma_layers = []
                for sma in ["SMA20", "SMA50", "SMA200"]:
                    if sma in chart_df.columns:
                        tmp = chart_df[["Datum", sma]].dropna().copy()
                        tmp["SMA_Type"] = sma
                        sma_layers.append(
                            alt.Chart(tmp).mark_line(strokeWidth=2).encode(
                                x="Datum:T",
                                y=alt.Y(f"{sma}:Q", title="Preis"),
                                color=alt.Color(
                                    "SMA_Type:N",
                                    scale=alt.Scale(domain=list(SMA_COLORS.keys()), range=list(SMA_COLORS.values())),
                                    legend=alt.Legend(title="Gleitende Durchschnitte", orient="bottom"),
                                ),
                                tooltip=["Datum", sma],
                            )
                        )

                final_chart = alt.layer(base_chart, *sma_layers).resolve_scale(y="shared").properties(height=300)
                st.altair_chart(final_chart, use_container_width=True)
            else:
                st.info("Für diesen Ticker ist keine Kurs-Historie verfügbar.")
    else:
        st.info("Bitte zuerst oben auf **'Daten laden / aktualisieren'** klicken.")

    # --- Graph-Ansicht ---
    st.subheader("Schritt 4: Graph-Ansicht (Mindmap-light)")
    if df_view.empty:
        st.info("Keine Firmen für diese Filtereinstellung gefunden.")
    else:
        G = build_graph(df_view)
        fig = draw_graph(G)
        if fig is not None:
            st.pyplot(fig)
        else:
            st.info("Graph konnte nicht gezeichnet werden (keine Knoten).")


if __name__ == "__main__":
    main()
