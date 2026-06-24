"""
stockdata_features.py
=====================

Versão estendida do `stockdata.py`:
  catálogo de ações + extrator OHLCV + cálculo de FEATURES.

Replica a estrutura do `stockdata.py` original (catálogo, extrator com cache e
download paralelo) e incorpora todo o pipeline de engenharia de variáveis:

    - Retornos (log e simples)
    - Fatores técnicos (RSI, MACD, Bollinger %B, ATR, OBV, VWAP-rel)
    - Factor zoo (momentum, volatility, liquidity)
    - Fatores macro diários (Selic, Fama-French 5, VIX, BRL/USD, DXY)
    - Betas rolling do retorno do ativo contra fatores macro

Entrega
-------
Um DataFrame "painel rico" (long), com colunas:

    date | ticker | unique_id | Open High Low Close Volume |
    ret_log | rsi_14 | macd_hist | ... | mkt_rf | vix | ...

Pronto para ser consumido pelo `transformador_features_nixtla.py`, que converte
para o formato canônico Nixtla.

Uso mínimo
----------
    from stockdata_features import StockFeaturesExtractor

    ext = StockFeaturesExtractor()
    painel = ext.gerar_painel_rico(start_date="2020-01-01")
    painel.to_parquet("painel_rico.parquet", index=False)
"""

from __future__ import annotations

import json
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# 1. ENUMS E DATACLASSES (mesmos do stockdata.py original)
# ============================================================================

class Regiao(Enum):
    BRASIL = "brasil"
    EUA    = "eua"
    EUROPA = "europa"
    ASIA   = "asia"


class Setor(Enum):
    FINANCEIRO       = "financeiro"
    ENERGIA          = "energia"
    MATERIAIS        = "materiais"
    CONSUMO_CICLICO  = "consumo_ciclico"
    CONSUMO_BASICO   = "consumo_basico"
    SAUDE            = "saude"
    INDUSTRIAL       = "industrial"
    TECNOLOGIA       = "tecnologia"
    TELECOM          = "telecom"
    UTILIDADES       = "utilidades"
    IMOBILIARIO      = "imobiliario"


@dataclass
class AcaoConfig:
    ticker: str
    nome: str
    regiao: Regiao
    setor: Setor
    pais: str
    descricao: str = ""
    moeda: str = "USD"


# ============================================================================
# 2. CATÁLOGO DE AÇÕES — subset representativo (~100 nomes)
# ============================================================================
# Estrutura idêntica à do stockdata.py original. Para reaproveitar um catálogo
# maior já existente, use `carregar_catalogo_externo(caminho)` mais abaixo.

CATALOGO_ACOES: Dict[str, AcaoConfig] = {
    # ---------- BRASIL ----------
    "itau":         AcaoConfig("ITUB4.SA", "Itaú",        Regiao.BRASIL, Setor.FINANCEIRO,      "Brasil", moeda="BRL"),
    "bradesco":     AcaoConfig("BBDC4.SA", "Bradesco",    Regiao.BRASIL, Setor.FINANCEIRO,      "Brasil", moeda="BRL"),
    "banco_brasil": AcaoConfig("BBAS3.SA", "BB",          Regiao.BRASIL, Setor.FINANCEIRO,      "Brasil", moeda="BRL"),
    "btg_pactual":  AcaoConfig("BPAC11.SA","BTG",         Regiao.BRASIL, Setor.FINANCEIRO,      "Brasil", moeda="BRL"),
    "b3":           AcaoConfig("B3SA3.SA", "B3",          Regiao.BRASIL, Setor.FINANCEIRO,      "Brasil", moeda="BRL"),
    "itausa":       AcaoConfig("ITSA4.SA", "Itaúsa",      Regiao.BRASIL, Setor.FINANCEIRO,      "Brasil", moeda="BRL"),
    "petrobras":    AcaoConfig("PETR4.SA", "Petrobras PN",Regiao.BRASIL, Setor.ENERGIA,         "Brasil", moeda="BRL"),
    "prio":         AcaoConfig("PRIO3.SA", "PRIO",        Regiao.BRASIL, Setor.ENERGIA,         "Brasil", moeda="BRL"),
    "cosan":        AcaoConfig("CSAN3.SA", "Cosan",       Regiao.BRASIL, Setor.ENERGIA,         "Brasil", moeda="BRL"),
    "ultrapar":     AcaoConfig("UGPA3.SA", "Ultrapar",    Regiao.BRASIL, Setor.ENERGIA,         "Brasil", moeda="BRL"),
    "vale":         AcaoConfig("VALE3.SA", "Vale",        Regiao.BRASIL, Setor.MATERIAIS,       "Brasil", moeda="BRL"),
    "csn":          AcaoConfig("CSNA3.SA", "CSN",         Regiao.BRASIL, Setor.MATERIAIS,       "Brasil", moeda="BRL"),
    "gerdau":       AcaoConfig("GGBR4.SA", "Gerdau",      Regiao.BRASIL, Setor.MATERIAIS,       "Brasil", moeda="BRL"),
    "suzano":       AcaoConfig("SUZB3.SA", "Suzano",      Regiao.BRASIL, Setor.MATERIAIS,       "Brasil", moeda="BRL"),
    "klabin":       AcaoConfig("KLBN11.SA","Klabin",      Regiao.BRASIL, Setor.MATERIAIS,       "Brasil", moeda="BRL"),
    "ambev":        AcaoConfig("ABEV3.SA", "Ambev",       Regiao.BRASIL, Setor.CONSUMO_BASICO,  "Brasil", moeda="BRL"),
    "jbs":          AcaoConfig("JBSS3.SA", "JBS",         Regiao.BRASIL, Setor.CONSUMO_BASICO,  "Brasil", moeda="BRL"),
    "assai":        AcaoConfig("ASAI3.SA", "Assaí",       Regiao.BRASIL, Setor.CONSUMO_BASICO,  "Brasil", moeda="BRL"),
    "raia_drogasil":AcaoConfig("RADL3.SA", "Raia Drogasil",Regiao.BRASIL,Setor.CONSUMO_BASICO,  "Brasil", moeda="BRL"),
    "localiza":     AcaoConfig("RENT3.SA", "Localiza",    Regiao.BRASIL, Setor.CONSUMO_CICLICO, "Brasil", moeda="BRL"),
    "magazine_luiza":AcaoConfig("MGLU3.SA","Magalu",      Regiao.BRASIL, Setor.CONSUMO_CICLICO, "Brasil", moeda="BRL"),
    "lojas_renner": AcaoConfig("LREN3.SA", "Renner",      Regiao.BRASIL, Setor.CONSUMO_CICLICO, "Brasil", moeda="BRL"),
    "weg":          AcaoConfig("WEGE3.SA", "WEG",         Regiao.BRASIL, Setor.INDUSTRIAL,      "Brasil", moeda="BRL"),
    "rumo":         AcaoConfig("RAIL3.SA", "Rumo",        Regiao.BRASIL, Setor.INDUSTRIAL,      "Brasil", moeda="BRL"),
    "totvs":        AcaoConfig("TOTS3.SA", "Totvs",       Regiao.BRASIL, Setor.TECNOLOGIA,      "Brasil", moeda="BRL"),
    "rede_dor":     AcaoConfig("RDOR3.SA", "Rede D'Or",   Regiao.BRASIL, Setor.SAUDE,           "Brasil", moeda="BRL"),
    "hapvida":      AcaoConfig("HAPV3.SA", "Hapvida",     Regiao.BRASIL, Setor.SAUDE,           "Brasil", moeda="BRL"),
    "telefonica_br":AcaoConfig("VIVT3.SA", "Vivo",        Regiao.BRASIL, Setor.TELECOM,         "Brasil", moeda="BRL"),
    "engie_brasil": AcaoConfig("EGIE3.SA", "Engie Brasil",Regiao.BRASIL, Setor.UTILIDADES,      "Brasil", moeda="BRL"),
    "sabesp":       AcaoConfig("SBSP3.SA", "Sabesp",      Regiao.BRASIL, Setor.UTILIDADES,      "Brasil", moeda="BRL"),

    # ---------- EUA ----------
    "apple":     AcaoConfig("AAPL", "Apple",      Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "microsoft": AcaoConfig("MSFT", "Microsoft",  Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "google":    AcaoConfig("GOOGL","Alphabet",   Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "amazon":    AcaoConfig("AMZN", "Amazon",     Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "nvidia":    AcaoConfig("NVDA", "Nvidia",     Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "meta":      AcaoConfig("META", "Meta",       Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "tesla":     AcaoConfig("TSLA", "Tesla",      Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "amd":       AcaoConfig("AMD",  "AMD",        Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "broadcom":  AcaoConfig("AVGO", "Broadcom",   Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "oracle":    AcaoConfig("ORCL", "Oracle",     Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "adobe":     AcaoConfig("ADBE", "Adobe",      Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "salesforce":AcaoConfig("CRM",  "Salesforce", Regiao.EUA, Setor.TECNOLOGIA,      "EUA"),
    "jpmorgan":  AcaoConfig("JPM",  "JPMorgan",   Regiao.EUA, Setor.FINANCEIRO,      "EUA"),
    "bank_of_america":AcaoConfig("BAC","BofA",    Regiao.EUA, Setor.FINANCEIRO,      "EUA"),
    "goldman_sachs":AcaoConfig("GS","Goldman",    Regiao.EUA, Setor.FINANCEIRO,      "EUA"),
    "berkshire": AcaoConfig("BRK-B","Berkshire",  Regiao.EUA, Setor.FINANCEIRO,      "EUA"),
    "visa":      AcaoConfig("V",    "Visa",       Regiao.EUA, Setor.FINANCEIRO,      "EUA"),
    "mastercard":AcaoConfig("MA",   "Mastercard", Regiao.EUA, Setor.FINANCEIRO,      "EUA"),
    "blackrock": AcaoConfig("BLK",  "BlackRock",  Regiao.EUA, Setor.FINANCEIRO,      "EUA"),
    "unitedhealth":AcaoConfig("UNH","UnitedHealth",Regiao.EUA,Setor.SAUDE,           "EUA"),
    "johnson_johnson":AcaoConfig("JNJ","JNJ",     Regiao.EUA, Setor.SAUDE,           "EUA"),
    "eli_lilly": AcaoConfig("LLY",  "Eli Lilly",  Regiao.EUA, Setor.SAUDE,           "EUA"),
    "pfizer":    AcaoConfig("PFE",  "Pfizer",     Regiao.EUA, Setor.SAUDE,           "EUA"),
    "merck":     AcaoConfig("MRK",  "Merck",      Regiao.EUA, Setor.SAUDE,           "EUA"),
    "walmart":   AcaoConfig("WMT",  "Walmart",    Regiao.EUA, Setor.CONSUMO_BASICO,  "EUA"),
    "costco":    AcaoConfig("COST", "Costco",     Regiao.EUA, Setor.CONSUMO_BASICO,  "EUA"),
    "coca_cola": AcaoConfig("KO",   "Coca-Cola",  Regiao.EUA, Setor.CONSUMO_BASICO,  "EUA"),
    "pepsico":   AcaoConfig("PEP",  "PepsiCo",    Regiao.EUA, Setor.CONSUMO_BASICO,  "EUA"),
    "home_depot":AcaoConfig("HD",   "Home Depot", Regiao.EUA, Setor.CONSUMO_CICLICO, "EUA"),
    "mcdonalds": AcaoConfig("MCD",  "McDonald's", Regiao.EUA, Setor.CONSUMO_CICLICO, "EUA"),
    "nike":      AcaoConfig("NKE",  "Nike",       Regiao.EUA, Setor.CONSUMO_CICLICO, "EUA"),
    "exxon":     AcaoConfig("XOM",  "ExxonMobil", Regiao.EUA, Setor.ENERGIA,         "EUA"),
    "chevron":   AcaoConfig("CVX",  "Chevron",    Regiao.EUA, Setor.ENERGIA,         "EUA"),
    "caterpillar":AcaoConfig("CAT", "Caterpillar",Regiao.EUA, Setor.INDUSTRIAL,      "EUA"),
    "boeing":    AcaoConfig("BA",   "Boeing",     Regiao.EUA, Setor.INDUSTRIAL,      "EUA"),
    "netflix":   AcaoConfig("NFLX", "Netflix",    Regiao.EUA, Setor.TELECOM,         "EUA"),
    "disney":    AcaoConfig("DIS",  "Disney",     Regiao.EUA, Setor.TELECOM,         "EUA"),

    # ---------- EUROPA ----------
    "asml":        AcaoConfig("ASML",   "ASML",        Regiao.EUROPA, Setor.TECNOLOGIA,      "Holanda"),
    "sap":         AcaoConfig("SAP",    "SAP",         Regiao.EUROPA, Setor.TECNOLOGIA,      "Alemanha"),
    "hsbc":        AcaoConfig("HSBC",   "HSBC",        Regiao.EUROPA, Setor.FINANCEIRO,      "Reino Unido"),
    "ubs":         AcaoConfig("UBS",    "UBS",         Regiao.EUROPA, Setor.FINANCEIRO,      "Suíça"),
    "santander":   AcaoConfig("SAN",    "Santander",   Regiao.EUROPA, Setor.FINANCEIRO,      "Espanha"),
    "shell":       AcaoConfig("SHEL",   "Shell",       Regiao.EUROPA, Setor.ENERGIA,         "Reino Unido"),
    "bp":          AcaoConfig("BP",     "BP",          Regiao.EUROPA, Setor.ENERGIA,         "Reino Unido"),
    "totalenergies":AcaoConfig("TTE",   "TotalEnergies",Regiao.EUROPA,Setor.ENERGIA,         "França"),
    "rio_tinto":   AcaoConfig("RIO",    "Rio Tinto",   Regiao.EUROPA, Setor.MATERIAIS,       "Reino Unido"),
    "bhp":         AcaoConfig("BHP",    "BHP",         Regiao.EUROPA, Setor.MATERIAIS,       "Austrália"),
    "arcelormittal":AcaoConfig("MT",    "ArcelorMittal",Regiao.EUROPA,Setor.MATERIAIS,       "Luxemburgo"),
    "linde":       AcaoConfig("LIN",    "Linde",       Regiao.EUROPA, Setor.MATERIAIS,       "Irlanda"),
    "nestle":      AcaoConfig("NSRGY",  "Nestlé",      Regiao.EUROPA, Setor.CONSUMO_BASICO,  "Suíça"),
    "unilever":    AcaoConfig("UL",     "Unilever",    Regiao.EUROPA, Setor.CONSUMO_BASICO,  "Reino Unido"),
    "ab_inbev":    AcaoConfig("BUD",    "AB InBev",    Regiao.EUROPA, Setor.CONSUMO_BASICO,  "Bélgica"),
    "novartis":    AcaoConfig("NVS",    "Novartis",    Regiao.EUROPA, Setor.SAUDE,           "Suíça"),
    "roche":       AcaoConfig("RHHBY",  "Roche",       Regiao.EUROPA, Setor.SAUDE,           "Suíça"),
    "novo_nordisk":AcaoConfig("NVO",    "Novo Nordisk",Regiao.EUROPA, Setor.SAUDE,           "Dinamarca"),
    "astrazeneca": AcaoConfig("AZN",    "AstraZeneca", Regiao.EUROPA, Setor.SAUDE,           "Reino Unido"),
    "ferrari":     AcaoConfig("RACE",   "Ferrari",     Regiao.EUROPA, Setor.CONSUMO_CICLICO, "Itália"),
    "spotify":     AcaoConfig("SPOT",   "Spotify",     Regiao.EUROPA, Setor.TECNOLOGIA,      "Suécia"),

    # ---------- ÁSIA ----------
    "tsmc":             AcaoConfig("TSM",    "TSMC",            Regiao.ASIA, Setor.TECNOLOGIA, "Taiwan"),
    "samsung":          AcaoConfig("005930.KS","Samsung",       Regiao.ASIA, Setor.TECNOLOGIA, "Coreia", moeda="KRW"),
    "sony":             AcaoConfig("SONY",   "Sony",            Regiao.ASIA, Setor.TECNOLOGIA, "Japão"),
    "toyota":           AcaoConfig("TM",     "Toyota",          Regiao.ASIA, Setor.CONSUMO_CICLICO,"Japão"),
    "honda":            AcaoConfig("HMC",    "Honda",           Regiao.ASIA, Setor.CONSUMO_CICLICO,"Japão"),
    "alibaba":          AcaoConfig("BABA",   "Alibaba",         Regiao.ASIA, Setor.TECNOLOGIA, "China"),
    "tencent":          AcaoConfig("TCEHY",  "Tencent",         Regiao.ASIA, Setor.TECNOLOGIA, "China"),
    "jd_com":           AcaoConfig("JD",     "JD.com",          Regiao.ASIA, Setor.TECNOLOGIA, "China"),
    "pinduoduo":        AcaoConfig("PDD",    "Pinduoduo",       Regiao.ASIA, Setor.TECNOLOGIA, "China"),
    "byd":              AcaoConfig("BYDDY",  "BYD",             Regiao.ASIA, Setor.CONSUMO_CICLICO,"China"),
    "nio":              AcaoConfig("NIO",    "NIO",             Regiao.ASIA, Setor.CONSUMO_CICLICO,"China"),
    "infosys":          AcaoConfig("INFY",   "Infosys",         Regiao.ASIA, Setor.TECNOLOGIA, "Índia"),
    "hdfc_bank":        AcaoConfig("HDB",    "HDFC Bank",       Regiao.ASIA, Setor.FINANCEIRO, "Índia"),
    "icici_bank":       AcaoConfig("IBN",    "ICICI Bank",      Regiao.ASIA, Setor.FINANCEIRO, "Índia"),
}


def carregar_catalogo_externo(caminho: Union[str, Path]) -> Dict[str, AcaoConfig]:
    """Carrega um `CATALOGO_ACOES` definido em outro arquivo `.py`, sem disparar
    código de execução residual (corta no primeiro `if __name__ == "__main__":`)."""
    import re
    src = Path(caminho).read_text(encoding="utf-8")
    m = re.search(r'^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', src, re.M)
    if m:
        src = src[: m.start()]
    ns: Dict[str, object] = {"__name__": "catalogo_externo", "__file__": str(caminho)}
    exec(compile(src, str(caminho), "exec"), ns)
    if "CATALOGO_ACOES" not in ns:
        raise RuntimeError(f"CATALOGO_ACOES não encontrado em {caminho}.")
    return ns["CATALOGO_ACOES"]


# ============================================================================
# 3. EXTRATOR — download + cache + features
# ============================================================================

@dataclass
class ExtractorConfig:
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    max_workers: int = 10
    cache_dir: str = "./cache_stocks_features"
    auto_adjust: bool = True
    # janelas (em dias úteis)
    window_rsi: int       = 14
    window_macd_fast: int = 12
    window_macd_slow: int = 26
    window_macd_signal:int = 9
    window_bb: int        = 20
    window_atr: int       = 14
    window_vol_curta: int = 21
    window_vol_media: int = 63
    window_vol_longa: int = 252
    window_beta: int      = 63
    fatores_beta: tuple   = ("mkt_rf", "vix", "brl_ret")


class StockFeaturesExtractor:
    """Extrator com catálogo + download paralelo + engenharia de features."""

    def __init__(self,
                 catalogo: Optional[Dict[str, AcaoConfig]] = None,
                 cfg: Optional[ExtractorConfig] = None):
        self.catalogo = catalogo if catalogo is not None else CATALOGO_ACOES
        self.cfg = cfg or ExtractorConfig()
        Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        self._yf = None

    # ---------- yfinance lazy import ----------
    def _yfinance(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    # ---------- listagens ----------
    def listar_por_regiao(self, regiao: Union[str, Regiao]) -> List[str]:
        if isinstance(regiao, str): regiao = Regiao(regiao)
        return [k for k, v in self.catalogo.items() if v.regiao == regiao]

    def listar_por_setor(self, setor: Union[str, Setor]) -> List[str]:
        if isinstance(setor, str): setor = Setor(setor)
        return [k for k, v in self.catalogo.items() if v.setor == setor]

    # ------------------------------------------------------------------
    # 3a. DOWNLOAD OHLCV
    # ------------------------------------------------------------------
    @staticmethod
    def _ticker_to_uid(ticker: str) -> str:
        """Extrai símbolo limpo: 'PETR4.SA' → 'PETR4', 'AAPL' → 'AAPL'."""
        return ticker.split(".")[0]

    # campos OHLCV reconhecidos (usados para achatar o MultiIndex com segurança)
    _OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

    @classmethod
    def _flatten_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Achata um MultiIndex de colunas escolhendo, em cada coluna, o nível
        que corresponde a um campo OHLCV — independente da ordem (campo,ticker)
        ou (ticker,campo) que o yfinance devolver."""
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        campos = set(cls._OHLCV_FIELDS)
        novas = []
        for tup in df.columns:
            match = [x for x in tup if x in campos]
            novas.append(match[0] if match else tup[-1])
        df = df.copy()
        df.columns = novas
        return df

    @classmethod
    def _normalizar_ohlcv(cls, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = cls._flatten_columns(df)
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")]
        df = df[df.index.dayofweek < 5]
        df = df.sort_index()
        df.index.name = "date"
        # remove colunas duplicadas (origem do InvalidIndexError no concat)
        df = df.loc[:, ~df.columns.duplicated()]
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if not keep:
            return pd.DataFrame()
        df = df[keep]
        df["ticker"] = ticker
        df["unique_id"] = cls._ticker_to_uid(ticker)
        return df

    def extrair_acao(self, acao_id: str) -> pd.DataFrame:
        if acao_id not in self.catalogo:
            raise ValueError(f"Ação '{acao_id}' não está no catálogo")
        cfg = self.cfg
        yf = self._yfinance()
        end_date = cfg.end_date or datetime.now().strftime("%Y-%m-%d")
        try:
            df = yf.download(self.catalogo[acao_id].ticker,
                             start=cfg.start_date, end=end_date,
                             auto_adjust=cfg.auto_adjust, progress=False, timeout=20)
            return self._normalizar_ohlcv(df, self.catalogo[acao_id].ticker)
        except Exception as e:
            warnings.warn(f"falha {acao_id}: {e}")
            return pd.DataFrame()

    def extrair_ohlcv(self, acoes: Optional[List[str]] = None,
                      verbose: bool = True) -> pd.DataFrame:
        """Baixa OHLCV em paralelo para a lista (None = catálogo todo). Cache em parquet."""
        cfg = self.cfg
        acoes = acoes or list(self.catalogo.keys())
        cache_file = Path(cfg.cache_dir) / f"ohlcv_{cfg.start_date}_{cfg.end_date or 'today'}_{len(acoes)}.parquet"
        if cache_file.exists():
            if verbose: print(f"  [cache] {cache_file}")
            return pd.read_parquet(cache_file)

        if verbose: print(f"  baixando {len(acoes)} ações...")

        def _baixa(a):
            return a, self.extrair_acao(a)

        resultados: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = {ex.submit(_baixa, a): a for a in acoes}
            for i, f in enumerate(as_completed(futs), 1):
                a, df = f.result()
                ok = (df is not None) and (not df.empty) and len(df) >= 30
                if ok: resultados[a] = df
                if verbose:
                    print(f"    [{i:3}/{len(acoes)}] {a}: {'OK' if ok else 'SEM_DADOS'}")

        if not resultados:
            raise RuntimeError("nenhuma ação baixada")
        panel = (pd.concat(resultados.values())
                   .reset_index()
                   .sort_values(["unique_id", "date"])
                   .reset_index(drop=True)
                   .set_index("date"))
        panel.to_parquet(cache_file)
        if verbose:
            print(f"  salvo: {cache_file}  ({len(panel):,} linhas, {panel['unique_id'].nunique()} ativos)")
        return panel

    # ------------------------------------------------------------------
    # 3b. PIVOT — long -> wide por campo
    # ------------------------------------------------------------------
    @staticmethod
    def _pivot(panel: pd.DataFrame, field: str) -> pd.DataFrame:
        return panel.reset_index().pivot_table(
            index="date", columns="unique_id", values=field, aggfunc="last"
        )

    # ------------------------------------------------------------------
    # 3c. RETORNOS
    # ------------------------------------------------------------------
    @staticmethod
    def calcular_retornos(close_w: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {
            "ret_log": np.log(close_w / close_w.shift(1)),
            "ret_simples": close_w.pct_change(),
        }

    # ------------------------------------------------------------------
    # 3d. FATORES TÉCNICOS (RSI/MACD/Bollinger/ATR/OBV/VWAP-rel)
    # ------------------------------------------------------------------
    def fatores_tecnicos(self, open_w, high_w, low_w, close_w, volume_w) -> Dict[str, pd.DataFrame]:
        cfg = self.cfg
        # RSI
        delta = close_w.diff()
        up   = delta.clip(lower=0).rolling(cfg.window_rsi).mean()
        down = (-delta.clip(upper=0)).rolling(cfg.window_rsi).mean()
        rs = up / down.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        # MACD
        ema_f = close_w.ewm(span=cfg.window_macd_fast, adjust=False).mean()
        ema_s = close_w.ewm(span=cfg.window_macd_slow, adjust=False).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=cfg.window_macd_signal, adjust=False).mean()
        macd_hist = macd - signal

        # Bollinger %B
        ma  = close_w.rolling(cfg.window_bb).mean()
        sd  = close_w.rolling(cfg.window_bb).std()
        bb_pct = (close_w - (ma - 2 * sd)) / (4 * sd + 1e-10)

        # ATR(14) normalizado
        tr = pd.concat([
            (high_w - low_w),
            (high_w - close_w.shift(1)).abs(),
            (low_w  - close_w.shift(1)).abs(),
        ]).groupby(level=0).max()
        atr = tr.rolling(cfg.window_atr).mean() / close_w

        # OBV z-score
        sign = np.sign(close_w.diff()).fillna(0)
        obv  = (sign * volume_w.fillna(0)).cumsum()
        obv_z = (obv - obv.rolling(cfg.window_vol_media).mean()) / (obv.rolling(cfg.window_vol_media).std() + 1e-10)

        # VWAP relative
        typical = (high_w + low_w + close_w) / 3
        vwap = (typical * volume_w).rolling(cfg.window_vol_curta).sum() / volume_w.rolling(cfg.window_vol_curta).sum()
        vwap_rel = close_w / vwap - 1

        return {
            "rsi_14":     rsi,
            "macd_hist":  macd_hist,
            "bb_pct":     bb_pct,
            "atr_norm":   atr,
            "obv_z63":    obv_z,
            "vwap_rel21": vwap_rel,
        }

    # ------------------------------------------------------------------
    # 3e. FACTOR ZOO
    # ------------------------------------------------------------------
    def factor_zoo(self, close_w: pd.DataFrame, volume_w: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ret = close_w.pct_change()
        out: Dict[str, pd.DataFrame] = {
            # momentum
            "mom_252_21":  ret.rolling(252).sum() - ret.rolling(21).sum(),
            "mom_126_21":  ret.rolling(126).sum() - ret.rolling(21).sum(),
            "mom_63_21":   ret.rolling(63).sum()  - ret.rolling(21).sum(),
            "reversal_21": -ret.rolling(21).sum(),
            "high_252w":   close_w / close_w.rolling(252).max(),
            # volatility
            "rvol_63":     ret.rolling(63).std() * np.sqrt(252),
            "rvol_21":     ret.rolling(21).std() * np.sqrt(252),
            "skew_63":     ret.rolling(63).skew(),
            "kurt_63":     ret.rolling(63).kurt(),
            "max_ret_21":  ret.rolling(21).max(),
            "min_ret_21":  ret.rolling(21).min(),
        }
        # liquidity (depende de volume)
        ret_abs    = ret.abs()
        dollar_vol = (close_w * volume_w).replace(0, np.nan)
        out["amihud_21"]   = (ret_abs / dollar_vol * 1e9).rolling(21).mean()
        out["turnover_21"] = np.log1p(volume_w.rolling(21).sum())
        return out

    # ------------------------------------------------------------------
    # 3f. FATORES MACRO
    # ------------------------------------------------------------------
    def macro_selic(self, start: str, end: str) -> pd.Series:
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
               f"?formato=json&dataInicial={pd.Timestamp(start).strftime('%d/%m/%Y')}"
               f"&dataFinal={pd.Timestamp(end).strftime('%d/%m/%Y')}")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
            df = pd.DataFrame(data)
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce") / 100
            return df.set_index("data")["valor"].rename("rf_br")
        except Exception as e:
            warnings.warn(f"BCB falhou ({e}) — proxy constante")
            idx = pd.bdate_range(start, end)
            return pd.Series(0.1375 / 252, index=idx, name="rf_br")

    def macro_fama_french(self, start: str, end: str) -> pd.DataFrame:
        try:
            import pandas_datareader.data as web
            ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                                "famafrench", start=start, end=end)[0] / 100
            ff.index = pd.to_datetime(ff.index)
            ff.index.name = "date"
            return ff.rename(columns={"Mkt-RF":"mkt_rf","SMB":"smb","HML":"hml",
                                       "RMW":"rmw","CMA":"cma","RF":"rf_us"})
        except Exception as e:
            warnings.warn(f"Fama-French falhou ({e}) — proxy zero")
            idx = pd.bdate_range(start, end)
            return pd.DataFrame({c: 0.0 for c in
                                ["mkt_rf","smb","hml","rmw","cma","rf_us"]}, index=idx)

    def macro_yfinance(self, start: str, end: str) -> pd.DataFrame:
        yf = self._yfinance()
        def _dl(tk, col, retorno):
            try:
                df = yf.download(tk, start=start, end=end,
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                s = df["Close"]
                if s.index.tz is not None: s.index = s.index.tz_localize(None)
                s.index = s.index.normalize()
                s = s[s.index.dayofweek < 5]
                return (s.pct_change() if retorno else s).rename(col)
            except Exception as e:
                warnings.warn(f"yf {tk}: {e}")
                return pd.Series(dtype=float, name=col)
        df = pd.concat([
            _dl("^VIX",     "vix",     False),
            _dl("BRL=X",    "brl_ret", True),
            _dl("DX-Y.NYB", "dxy_ret", True),
        ], axis=1).sort_index()
        df.index.name = "date"
        return df

    def montar_macro(self, start: str, end: str) -> pd.DataFrame:
        m = pd.concat([
            self.macro_selic(start, end).to_frame(),
            self.macro_fama_french(start, end),
            self.macro_yfinance(start, end),
        ], axis=1).sort_index()
        m.index = pd.to_datetime(m.index).normalize()
        m = m[~m.index.duplicated(keep="last")]
        return m.ffill()

    # ------------------------------------------------------------------
    # 3g. BETAS ROLLING contra fatores macro
    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_beta(ret_w: pd.DataFrame, factor: pd.Series, window: int) -> pd.DataFrame:
        f = factor.reindex(ret_w.index)
        cov = ret_w.rolling(window).cov(f)
        var = f.rolling(window).var()
        return cov.divide(var.replace(0, np.nan), axis=0)

    def betas_macro(self, ret_w: pd.DataFrame, macro: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        out = {}
        for f in self.cfg.fatores_beta:
            if f in macro.columns:
                out[f"beta_{f}"] = self._rolling_beta(ret_w, macro[f], self.cfg.window_beta)
        return out

    # ==================================================================
    # 4. ORQUESTRADOR — gera painel rico (long)
    # ==================================================================
    def gerar_painel_rico(self, acoes: Optional[List[str]] = None,
                          verbose: bool = True) -> pd.DataFrame:
        """Faz tudo: OHLCV -> retornos -> técnicos -> zoo -> macro -> betas
        e devolve um painel LONG: date, ticker, unique_id, OHLCV, todas as features."""
        if verbose: print("[1/5] OHLCV...")
        panel = self.extrair_ohlcv(acoes, verbose=verbose)

        open_w  = self._pivot(panel, "Open")
        high_w  = self._pivot(panel, "High")
        low_w   = self._pivot(panel, "Low")
        close_w = self._pivot(panel, "Close")
        vol_w   = self._pivot(panel, "Volume").fillna(0.0)

        if verbose: print("[2/5] Retornos...")
        feats: Dict[str, pd.DataFrame] = {}
        feats.update(self.calcular_retornos(close_w))

        if verbose: print("[3/5] Fatores técnicos + factor zoo...")
        feats.update(self.fatores_tecnicos(open_w, high_w, low_w, close_w, vol_w))
        feats.update(self.factor_zoo(close_w, vol_w))

        if verbose: print("[4/5] Macro + betas rolling...")
        start = self.cfg.start_date
        end   = self.cfg.end_date or datetime.now().strftime("%Y-%m-%d")
        macro = self.montar_macro(start, end)
        feats.update(self.betas_macro(close_w.pct_change(), macro))

        if verbose: print("[5/5] Empilhando painel rico (long)...")
        # 5a) features por ação -> long
        long_parts = []
        for nome, df_w in feats.items():
            s = df_w.stack(future_stack=True).rename(nome).reset_index()
            long_parts.append(s.set_index(["date", "unique_id"]))
        feat_long = pd.concat(long_parts, axis=1).reset_index()

        # 5b) join com OHLCV (long original)
        ohlcv_long = panel.reset_index()
        painel = ohlcv_long.merge(feat_long, on=["date", "unique_id"], how="left")

        # 5c) broadcast do macro por data
        macro_long = macro.reset_index().rename(columns={"index": "date"})
        if "date" not in macro_long.columns:
            macro_long = macro_long.rename(columns={macro_long.columns[0]: "date"})
        painel = painel.merge(macro_long, on="date", how="left")

        # 5d) limpeza
        painel = painel.replace([np.inf, -np.inf], np.nan)
        painel = painel.sort_values(["unique_id", "date"]).reset_index(drop=True)

        if verbose:
            print(f"      shape={painel.shape}  ativos={painel['unique_id'].nunique()}")
            print(f"      período={painel['date'].min().date()} → {painel['date'].max().date()}")
        return painel


# ============================================================================
# 5. EXEMPLO
# ============================================================================

if __name__ == "__main__":
    ext = StockFeaturesExtractor(
        cfg=ExtractorConfig(start_date="2022-01-01",
                            cache_dir="./cache_stocks_features"),
    )
    painel = ext.gerar_painel_rico(verbose=True)
    painel.to_parquet("painel_rico.parquet", index=False)
    print("\n=== AMOSTRA ===")
    print(painel.head())
    print("\n=== COLUNAS ===")
    print(painel.columns.tolist())
