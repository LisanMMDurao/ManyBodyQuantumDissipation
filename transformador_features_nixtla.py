"""
transformador_features_nixtla.py
================================

Transformador que converte o "painel rico" produzido pelo
`stockdata_features.StockFeaturesExtractor.gerar_painel_rico()` para o
formato canônico Nixtla, incluindo variáveis exógenas históricas.

Inspirado em `transformadados.py` (pipeline carregar → tratar NaN → converter →
salvar), mas adaptado para um painel já em formato long com múltiplas features.

Formato Nixtla com exógenas:
    | unique_id | ds | y | <exog1> | <exog2> | ... |

`y` é, por padrão, o log-retorno (`ret_log`). Todas as demais features (técnicas,
factor zoo e macro) viram **variáveis exógenas históricas** (`hist_exog_list`).

Uso mínimo
----------
    from stockdata_features import StockFeaturesExtractor, ExtractorConfig
    from transformador_features_nixtla import NixtlaFeaturesTransformer, MetodoImputacao

    ext = StockFeaturesExtractor(cfg=ExtractorConfig(start_date="2022-01-01"))
    painel = ext.gerar_painel_rico()

    tf = NixtlaFeaturesTransformer(target="ret_log",
                                   exclude=("Open","High","Low","Close","Volume",
                                            "ret_simples","ticker"))
    Y_df = tf.processar(painel, metodo_nan=MetodoImputacao.FORWARD_FILL,
                        save_path="Y_df_stocks_nixtla.parquet")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

class MetodoImputacao(Enum):
    """Como tratar NaN nas colunas de features."""
    MEDIA          = "media"
    MEDIANA        = "mediana"
    FORWARD_FILL   = "ffill"     # ideal para macro + features acumuladas
    BACKWARD_FILL  = "bfill"
    INTERPOLACAO   = "interpolacao"
    ZERO           = "zero"
    NENHUM         = "nenhum"    # mantém NaN (modelo trata)


@dataclass
class InfoConversao:
    n_linhas: int
    n_ativos: int
    n_features: int
    target: str
    exog_cols: List[str]
    data_inicio: pd.Timestamp
    data_fim: pd.Timestamp
    n_nan_antes: int
    n_nan_depois: int


# ============================================================================
# FUNÇÕES BÁSICAS
# ============================================================================

def tratar_nan_grupo(df: pd.DataFrame,
                     colunas: List[str],
                     metodo: MetodoImputacao,
                     id_col: str = "unique_id") -> pd.DataFrame:
    """Trata NaN por SÉRIE (por unique_id), preservando a estrutura temporal."""
    if metodo == MetodoImputacao.NENHUM:
        return df

    df = df.copy()
    g = df.groupby(id_col, sort=False, group_keys=False)

    if metodo == MetodoImputacao.MEDIA:
        for c in colunas:
            df[c] = g[c].transform(lambda s: s.fillna(s.mean()))
    elif metodo == MetodoImputacao.MEDIANA:
        for c in colunas:
            df[c] = g[c].transform(lambda s: s.fillna(s.median()))
    elif metodo == MetodoImputacao.FORWARD_FILL:
        for c in colunas:
            df[c] = g[c].transform(lambda s: s.ffill().bfill())
    elif metodo == MetodoImputacao.BACKWARD_FILL:
        for c in colunas:
            df[c] = g[c].transform(lambda s: s.bfill().ffill())
    elif metodo == MetodoImputacao.INTERPOLACAO:
        for c in colunas:
            df[c] = g[c].transform(lambda s: s.interpolate(method="linear").ffill().bfill())
    elif metodo == MetodoImputacao.ZERO:
        df[colunas] = df[colunas].fillna(0)

    return df


# ============================================================================
# CLASSE PRINCIPAL
# ============================================================================

class NixtlaFeaturesTransformer:
    """Transforma um painel rico (long, com OHLCV+features+macro) para Nixtla.

    Parameters
    ----------
    target : str
        Coluna do painel que vira `y`. Padrão: 'ret_log'.
    date_col : str
        Coluna de data no painel (vira `ds`).
    id_col : str
        Coluna identificadora da série (vira `unique_id`).
    exclude : Iterable[str]
        Colunas a remover (ex.: 'ticker', preços brutos). Não entram como exógenas.
    include : Optional[Iterable[str]]
        Se passado, mantém apenas estas colunas como exógenas (sobrescreve exclude).
    """

    def __init__(self,
                 target: str = "ret_log",
                 date_col: str = "date",
                 id_col: str = "unique_id",
                 exclude: Iterable[str] = ("Open","High","Low","Close","Volume",
                                           "ret_simples","ticker"),
                 include: Optional[Iterable[str]] = None):
        self.target   = target
        self.date_col = date_col
        self.id_col   = id_col
        self.exclude  = set(exclude)
        self.include  = set(include) if include is not None else None

        self.df_painel: Optional[pd.DataFrame] = None
        self.df_nixtla: Optional[pd.DataFrame] = None
        self.info:      Optional[InfoConversao] = None

    # ------------------------------------------------------------------
    # 1. carregar
    # ------------------------------------------------------------------
    def carregar(self,
                 painel: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """Aceita um DataFrame ou um caminho para parquet/csv."""
        if isinstance(painel, (str, Path)):
            p = Path(painel)
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            elif p.suffix in (".csv", ".tsv"):
                df = pd.read_csv(p)
            else:
                raise ValueError(f"extensão não suportada: {p.suffix}")
        else:
            df = painel.copy()

        # validações
        for col in (self.date_col, self.id_col, self.target):
            if col not in df.columns:
                raise KeyError(f"coluna obrigatória ausente: '{col}'")

        # normaliza tipos
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.normalize()

        self.df_painel = df
        print(f"✓ carregado: {df.shape}  | ativos={df[self.id_col].nunique()}"
              f" | colunas={len(df.columns)}")
        return df

    # ------------------------------------------------------------------
    # 2. selecionar exógenas
    # ------------------------------------------------------------------
    def selecionar_exogenas(self) -> List[str]:
        if self.df_painel is None:
            raise RuntimeError("chame carregar() primeiro")

        cols_base = {self.date_col, self.id_col, self.target}

        if self.include is not None:
            exog = [c for c in self.include if c in self.df_painel.columns
                                            and c not in cols_base]
        else:
            exog = [c for c in self.df_painel.columns
                    if c not in cols_base and c not in self.exclude]

        # só numéricas
        num = self.df_painel[exog].select_dtypes(include=[np.number]).columns.tolist()
        ignored = sorted(set(exog) - set(num))
        if ignored:
            print(f"  exógenas não numéricas ignoradas: {ignored}")
        return num

    # ------------------------------------------------------------------
    # 3. tratar NaN
    # ------------------------------------------------------------------
    def tratar_nan(self,
                   metodo: Union[str, MetodoImputacao] = MetodoImputacao.FORWARD_FILL,
                   colunas: Optional[List[str]] = None) -> pd.DataFrame:
        if self.df_painel is None:
            raise RuntimeError("chame carregar() primeiro")
        if isinstance(metodo, str):
            metodo = MetodoImputacao(metodo)
        colunas = colunas or self.selecionar_exogenas()
        n_antes = self.df_painel[colunas].isna().sum().sum()
        self.df_painel = tratar_nan_grupo(self.df_painel, colunas, metodo, self.id_col)
        n_depois = self.df_painel[colunas].isna().sum().sum()
        self._n_nan_antes  = int(n_antes)
        self._n_nan_depois = int(n_depois)
        print(f"✓ NaN tratados: {n_antes:,} -> {n_depois:,}  (método={metodo.value})")
        return self.df_painel

    # ------------------------------------------------------------------
    # 4. converter para Nixtla
    # ------------------------------------------------------------------
    def converter_nixtla(self,
                         min_obs_por_serie: int = 60,
                         drop_target_nan: bool = True) -> pd.DataFrame:
        if self.df_painel is None:
            raise RuntimeError("chame carregar() primeiro")

        exog_cols = self.selecionar_exogenas()
        df = self.df_painel.copy()

        # renomeia para padrão Nixtla
        df = df.rename(columns={self.date_col: "ds", self.id_col: "unique_id"})
        df = df.rename(columns={self.target: "y"})

        # remove infinitos
        df = df.replace([np.inf, -np.inf], np.nan)

        # alvo não pode ser NaN
        if drop_target_nan:
            n0 = len(df)
            df = df.dropna(subset=["y"])
            if len(df) < n0:
                print(f"  - {n0 - len(df):,} linhas com y=NaN removidas")

        # mantém só colunas relevantes
        keep = ["unique_id", "ds", "y"] + exog_cols
        df = df[[c for c in keep if c in df.columns]]

        # filtra séries muito curtas
        cont = df.groupby("unique_id").size()
        ok   = cont[cont >= min_obs_por_serie].index
        n_antes_series = df["unique_id"].nunique()
        df = df[df["unique_id"].isin(ok)]
        n_depois_series = df["unique_id"].nunique()
        if n_depois_series < n_antes_series:
            print(f"  - {n_antes_series - n_depois_series} séries removidas "
                  f"(< {min_obs_por_serie} observações)")

        # ordena
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        self.df_nixtla = df
        self.info = InfoConversao(
            n_linhas    = len(df),
            n_ativos    = df["unique_id"].nunique(),
            n_features  = len(exog_cols),
            target      = self.target,
            exog_cols   = exog_cols,
            data_inicio = df["ds"].min(),
            data_fim    = df["ds"].max(),
            n_nan_antes = getattr(self, "_n_nan_antes", 0),
            n_nan_depois= getattr(self, "_n_nan_depois", 0),
        )
        print(f"✓ Nixtla: {df.shape}  | ativos={self.info.n_ativos}"
              f"  | exógenas={self.info.n_features}")
        return df

    # ------------------------------------------------------------------
    # 5. salvar
    # ------------------------------------------------------------------
    def salvar(self, caminho: Union[str, Path], formato: str = "parquet"):
        if self.df_nixtla is None:
            raise RuntimeError("execute converter_nixtla() ou processar()")
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        if formato == "parquet":
            self.df_nixtla.to_parquet(caminho, index=False)
        elif formato == "csv":
            self.df_nixtla.to_csv(caminho, index=False)
        else:
            raise ValueError(f"formato '{formato}' não suportado")
        print(f"✓ salvo: {caminho}")

    # ------------------------------------------------------------------
    # 6. pipeline orquestrador
    # ------------------------------------------------------------------
    def processar(self,
                  painel: Union[pd.DataFrame, str, Path],
                  metodo_nan: Union[str, MetodoImputacao] = MetodoImputacao.FORWARD_FILL,
                  min_obs_por_serie: int = 60,
                  save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        print("=" * 60)
        print("PAINEL RICO -> NIXTLA")
        print("=" * 60)
        print("\n[1/4] carregando painel...")
        self.carregar(painel)
        print("\n[2/4] selecionando exógenas...")
        exog = self.selecionar_exogenas()
        print(f"      {len(exog)} exógenas: {exog[:8]}{'...' if len(exog)>8 else ''}")
        print("\n[3/4] tratando NaN...")
        self.tratar_nan(metodo_nan)
        print("\n[4/4] convertendo para Nixtla...")
        self.converter_nixtla(min_obs_por_serie=min_obs_por_serie)
        if save_path is not None:
            print()
            self.salvar(save_path)
        print("\n" + "=" * 60)
        print("CONCLUÍDO")
        print("=" * 60)
        return self.df_nixtla

    # ------------------------------------------------------------------
    # 7. resumo / utilidades
    # ------------------------------------------------------------------
    def resumo(self) -> Dict[str, object]:
        if self.info is None:
            return {"status": "não processado"}
        return {
            "linhas":         self.info.n_linhas,
            "ativos":         self.info.n_ativos,
            "exogenas":       self.info.n_features,
            "target":         self.info.target,
            "periodo":        f"{self.info.data_inicio.date()} a {self.info.data_fim.date()}",
            "nan_antes":      self.info.n_nan_antes,
            "nan_depois":     self.info.n_nan_depois,
            "primeiras_exog": self.info.exog_cols[:10],
        }

    def hist_exog_list(self) -> List[str]:
        """Lista pronta para passar em `hist_exog_list` dos modelos NeuralForecast."""
        if self.info is None:
            raise RuntimeError("execute processar() primeiro")
        return list(self.info.exog_cols)


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Modo standalone — assume que o painel_rico.parquet já foi gerado
    # pelo stockdata_features.py
    from pathlib import Path

    PAINEL_PATH = Path("painel_rico.parquet")
    OUTPUT      = Path("Y_df_stocks_nixtla.parquet")

    if not PAINEL_PATH.exists():
        print(f"⚠ {PAINEL_PATH} não existe. Gerando via stockdata_features...")
        from stockdata_features import StockFeaturesExtractor, ExtractorConfig
        ext = StockFeaturesExtractor(
            cfg=ExtractorConfig(start_date="2022-01-01",
                                cache_dir="./cache_stocks_features")
        )
        painel = ext.gerar_painel_rico(verbose=True)
        painel.to_parquet(PAINEL_PATH, index=False)
        print(f"✓ painel rico salvo em {PAINEL_PATH}")

    tf = NixtlaFeaturesTransformer(
        target  = "ret_log",
        exclude = ("Open", "High", "Low", "Close", "Volume",
                   "ret_simples", "ticker"),
    )
    Y_df = tf.processar(
        PAINEL_PATH,
        metodo_nan        = MetodoImputacao.FORWARD_FILL,
        min_obs_por_serie = 60,
        save_path         = OUTPUT,
    )

    print("\n=== AMOSTRA ===")
    print(Y_df.head(8))
    print("\n=== RESUMO ===")
    for k, v in tf.resumo().items():
        print(f"  {k:18s}: {v}")
    print("\n=== hist_exog_list (p/ NeuralForecast) ===")
    print(tf.hist_exog_list())
