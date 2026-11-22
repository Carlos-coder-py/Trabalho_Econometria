"""
PRÉ-REQUISITO
    Ter rodado 02_tratamento.py e 03_exploracao.py (bases em dados/final/).

COMO RODAR (terminal do VS Code, na pasta raiz do projeto)
    python src/04_modelos.py

ESTRATÉGIA EMPÍRICA (resumo — detalhes nos comentários de cada bloco)
    • Estimador: Mínimos Quadrados Ordinários (MQO). Para variáveis 0/1,
      modelo de probabilidade linear (MPL).
    • Efeitos fixos (dummies): piloto, circuito (separado por período quando
      o traçado mudou) e equipe × temporada (desempenho do carro em cada ano,
      o que também absorve a mudança de regulamento de 2022 e de 2026).
      O efeito do clima é estimado, portanto, comparando o mesmo piloto, no
      mesmo circuito e com o mesmo carro em condições climáticas diferentes.
    • Erros-padrão agrupados (cluster) por corrida: pilotos da mesma corrida
      compartilham pista e clima, então suas observações não são
      independentes.
    • Sem "controles ruins": Safety Car, pneu de chuva e posição de largada
      NÃO entram como controles, pois são consequências do clima (ou da
      classificação feita sob o mesmo clima) e absorveriam parte do efeito.

    4.1 Carregar as bases e criar as variáveis dos modelos
    4.2 H1 — temperatura x ritmo e consistência (corridas secas)      Tabela M1
    4.3 H1 — modelo no nível da volta (robustez)                      Tabela M2
    4.4 H2 — chuva x cada tipo de erro (alternativa B)                Tabela M3
    4.5 H3 — experiência do piloto modera o efeito da chuva           Tabelas M4 e M4b
    4.5c H3 — análise complementar: diferenças piloto a piloto        Tabelas M4c e M4d
    4.6 Robustez de H1 e H2                                           Tabela M5
    4.7 Figuras dos modelos                                           Figuras 10–13

SAÍDAS
    resultados/tabelas/tabelas_modelos.xlsx  -> tabelas no formato de artigo
    resultados/tabelas/M*.csv                -> as mesmas tabelas em CSV
    resultados/figuras/F10–F13*.png          -> gráficos dos modelos
    resultados/resumo_modelos.txt            -> números-chave
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")


# =====================================================================
# 0. CONFIGURAÇÃO
# =====================================================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
DADOS_DIR = os.path.join(BASE_DIR, "dados")
TRATADOS_DIR = os.path.join(DADOS_DIR, "tratados")
FINAL_DIR = os.path.join(DADOS_DIR, "final")
RES_DIR = os.path.join(BASE_DIR, "resultados")
TAB_DIR = os.path.join(RES_DIR, "tabelas")
FIG_DIR = os.path.join(RES_DIR, "figuras")

# --- Decisões metodológicas desta etapa (citar na metodologia) ---
# Circuitos cujo traçado mudou dentro do período: o efeito fixo de circuito
# é separado em "antes" e "a partir de" o ano da mudança.
MUDANCAS_TRACADO = {
    "Barcelona": 2023,     # retirada da chicane final (curvas 14–15)
    "Marina Bay": 2023,    # retirada das curvas 16–19
    "Singapore": 2023,
}
MIN_VOLTAS_TAXA = 10        # taxas por 100 voltas só para quem disputou >= 10 voltas
LIMITES_VOLTA_LENTA_ROBUSTEZ = (8.0, 15.0)   # sensibilidade ao limite de 10%
ALFA = 0.05
MIN_MOLHADAS_H3_ROBUSTEZ = 8     # robustez de H3: pilotos com mais corridas molhadas
N_PERMUTACOES = 199              # sorteios do teste de permutação (H3 complementar)
PERCENTIS_EXPERIENCIA = (0.10, 0.90)   # "novato" e "veterano" ao reportar o efeito da chuva
SEMENTE = 2026                   # semente aleatória (reprodutibilidade)

COR_SECO, COR_MOLHADO = "#eb6834", "#2a78d6"
COR_NEUTRA, COR_TEXTO, COR_TEXTO2 = "#8a8985", "#0b0b0b", "#52514e"

TABELAS = []
RESUMO = []


def log(msg):
    print(msg, flush=True)
    RESUMO.append(msg)


# =====================================================================
# 0.1 FUNÇÕES AUXILIARES (leitura segura, regressão e tabelas)
# =====================================================================

def como_bool(s):
    return s.astype(str).str.strip().str.upper().isin(["TRUE", "1", "1.0"])


def a_numero(s):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    t = s.astype("string").str.strip().str.upper().replace({"TRUE": "1", "FALSE": "0"})
    return pd.to_numeric(t, errors="coerce").astype(float)


def estrelas(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_p(p):
    return "" if pd.isna(p) else ("<0,001" if p < 0.001 else f"{p:.3f}")


def remover_efeitos_fixos(df, colunas, efeitos, tol=1e-9, max_iter=1000):
    """Transformação "within" com vários efeitos fixos: retira de cada coluna
    as médias de cada grupo (piloto, circuito, equipe-temporada...), de forma
    alternada, até convergir. Pelo teorema de Frisch–Waugh–Lovell, regredir
    as variáveis transformadas equivale a incluir todas as dummies, mas é
    numericamente estável mesmo com muitas dummies."""
    X = df[colunas].astype(float).copy()
    grupos = [pd.factorize(df[f])[0] for f in efeitos]
    for _ in range(max_iter):
        antes = X.values.copy()
        for g in grupos:
            medias = X.groupby(g).transform("mean")
            X = X - medias.values
        if np.nanmax(np.abs(X.values - antes)) < tol:
            break
    return X


def mqo_cluster(formula, dados, cluster="RaceId"):
    # cluster: "RaceId" (padrão) ou ["RaceId", "Driver"] (agrupamento duplo,
    # usado na robustez de H3: corrida e piloto ao mesmo tempo).
    """MQO com efeitos fixos e erros-padrão agrupados por corrida.

    Lê a fórmula assim: termos C(var) são EFEITOS FIXOS (absorvidos pela
    transformação within, sem estimar cada dummy); termos C(var):z criam uma
    inclinação de z para cada nível de var (usado em H3); os demais termos
    são variáveis explicativas. Os graus de liberdade descontam os efeitos
    fixos absorvidos."""
    y_nome, lado_dir = [t.strip() for t in formula.split("~")]
    termos = [t.strip() for t in lado_dir.split("+")]
    efeitos = [t[2:-1] for t in termos if t.startswith("C(") and ":" not in t]
    interacoes = [t for t in termos if t.startswith("C(") and ":" in t]
    xs = [t for t in termos if not t.startswith("C(")]

    clusters = [cluster] if isinstance(cluster, str) else list(cluster)
    usadas = [y_nome] + clusters + efeitos + xs + [t.split(":")[1] for t in interacoes] \
        + [t.split(":")[0][2:-1] for t in interacoes]
    d = dados.dropna(subset=list(dict.fromkeys(usadas))).copy()
    cols = list(xs)
    for t in interacoes:                         # C(Driver):Molhada
        fator, z = t.split(":")[0][2:-1], t.split(":")[1]
        for nivel in sorted(d[fator].astype(str).unique()):
            nome = f"C({fator})[{nivel}]:{z}"
            d[nome] = (d[fator].astype(str) == nivel) * d[z].astype(float)
            cols.append(nome)
    # retira singletons (grupo de efeito fixo com uma só observação)
    for f in efeitos:
        d = d[d.groupby(f)[y_nome].transform("count") > 1]
    W = remover_efeitos_fixos(d, [y_nome] + cols, efeitos) if efeitos else d[[y_nome] + cols]
    # Guarda de identificação: uma variável que quase não varia depois de
    # retirar os efeitos fixos (ex.: chuva que nunca muda dentro do mesmo
    # circuito na subamostra) não é estimável e é descartada, em vez de
    # produzir um coeficiente absurdo. Também descarta colunas colineares.
    desvio_orig = d[cols].astype(float).std()
    ok = [c for c in cols if desvio_orig[c] > 0 and W[c].std() > 1e-6 * desvio_orig[c]]
    if ok:
        from scipy.linalg import qr
        _, Rq, piv = qr(W[ok].values, mode="economic", pivoting=True)
        tol = np.abs(Rq[0, 0]) * max(W[ok].shape) * np.finfo(float).eps * 1e4
        ok = [ok[i] for i in sorted(piv[:int((np.abs(np.diag(Rq)) > tol).sum())])]
    descartadas = [c for c in cols if c not in ok]
    cols = ok
    if len(clusters) == 1:
        grupos = pd.factorize(d[clusters[0]])[0]
        n_grupos = len(np.unique(grupos))
    else:   # agrupamento duplo (Cameron, Gelbach e Miller, 2011)
        grupos = np.column_stack([pd.factorize(d[c])[0] for c in clusters])
        n_grupos = min(len(np.unique(grupos[:, j])) for j in range(grupos.shape[1]))
    res = sm.OLS(W[y_nome], W[cols]).fit(cov_type="cluster", cov_kwds={"groups": grupos})
    # Correção de graus de liberdade pelos efeitos fixos absorvidos
    n, k = len(d), len(cols)
    k_ef = sum(d[f].nunique() - (1 if i else 0) for i, f in enumerate(efeitos))
    fator = (n - k) / (n - k - k_ef) if efeitos and n - k - k_ef > 0 else 1.0
    r = ResultadoEF(res, res.cov_params() * fator, n_grupos, d, y_nome)
    r.descartadas = descartadas
    return r


class ResultadoEF:
    """Resultado do MQO com efeitos fixos: coeficientes, erros-padrão
    agrupados (com correção de graus de liberdade) e testes. A inferência
    usa a distribuição t com (nº de corridas − 1) graus de liberdade."""

    def __init__(self, res, cov, n_clusters, dados, y_nome):
        self.res, self.cov = res, cov
        self.params = res.params
        self.bse = np.sqrt(pd.Series(np.diag(cov), index=cov.index))
        self.n_clusters, self.dados, self.y_nome = n_clusters, dados, y_nome
        self.gl = n_clusters - 1
        self.tvalues = self.params / self.bse
        self.pvalues = pd.Series(2 * stats.t.sf(np.abs(self.tvalues), self.gl), index=self.params.index)
        self.nobs = res.nobs
        self.rsquared = res.rsquared

    def cov_params(self):
        return self.cov

    def conf_int(self, alfa=0.05):
        tc = stats.t.ppf(1 - alfa / 2, self.gl)
        return pd.DataFrame({0: self.params - tc * self.bse, 1: self.params + tc * self.bse})

    def f_test(self, R):
        return self.res.f_test(R, cov_p=self.cov.values)


def poisson_cluster(formula, dados, exposicao, cluster="RaceId",
                    efeitos=("Driver", "CircuitoLayout", "TeamSeason")):
    """Regressão de Poisson (contagens) com exposição = nº de voltas e erros
    agrupados por corrida. Usada como robustez das taxas por 100 voltas.
    Como em estimadores de Poisson com efeitos fixos, (i) retira os grupos
    de efeito fixo em que a contagem é sempre zero (não informam nada e
    impedem a convergência) e (ii) retira colunas de dummies perfeitamente
    colineares (ex.: dupla de companheiros de equipe sempre juntos)."""
    from patsy import dmatrices
    from scipy.linalg import qr
    y_nome = formula.split("~")[0].strip()
    d = dados.dropna(subset=[exposicao, y_nome]).copy()
    d = d[d[exposicao] > 0]
    mudou = True
    while mudou:                                   # (i) grupos só com zeros
        n0 = len(d)
        for fe in efeitos:
            soma = d.groupby(fe)[y_nome].transform("sum")
            d = d[soma > 0]
        mudou = len(d) < n0
    y, X = dmatrices(formula, d, return_type="dataframe")
    _, R, piv = qr(X.values, mode="economic", pivoting=True)   # (ii) colinearidade
    tol = np.abs(R[0, 0]) * max(X.shape) * np.finfo(float).eps * 1e3
    posto = int((np.abs(np.diag(R)) > tol).sum())
    X = X.iloc[:, np.sort(piv[:posto])]
    grupos = pd.factorize(d.loc[X.index, cluster])[0]
    res = sm.GLM(y, X, family=sm.families.Poisson(),
                 offset=np.log(d.loc[X.index, exposicao])).fit(
        cov_type="cluster", cov_kwds={"groups": grupos})
    res.n_clusters = len(np.unique(grupos))
    return res


def coef(res, nome):
    """Coeficiente, erro-padrão e p-valor de uma variável (NaN se ausente)."""
    if nome not in res.params.index:
        return np.nan, np.nan, np.nan
    return res.params[nome], res.bse[nome], res.pvalues[nome]


def tabela_regressao(modelos, variaveis, rotulos, extras):
    """Monta uma tabela no formato de artigo: uma coluna por modelo,
    coeficiente com estrelas e erro-padrão entre parênteses abaixo."""
    linhas = []
    for v in variaveis:
        l1, l2 = {"Variável": rotulos.get(v, v)}, {"Variável": ""}
        for nome, res in modelos.items():
            b, ep, p = coef(res, v)
            l1[nome] = "" if pd.isna(b) else f"{b:.4f}{estrelas(p)}"
            l2[nome] = "" if pd.isna(ep) else f"({ep:.4f})"
        linhas += [l1, l2]
    for rotulo, func in extras:
        l = {"Variável": rotulo}
        for nome, res in modelos.items():
            l[nome] = func(res)
        linhas.append(l)
    return pd.DataFrame(linhas)


EXTRAS_PADRAO = [
    ("Observações", lambda r: f"{int(r.nobs)}"),
    ("R² (within)", lambda r: f"{r.rsquared:.3f}" if hasattr(r, "rsquared") else ""),
    ("Corridas (clusters)", lambda r: f"{r.n_clusters}"),
]


def registrar_tabela(aba, titulo, df, nota=""):
    TABELAS.append((aba, titulo, df, nota))
    df.to_csv(os.path.join(TAB_DIR, f"{aba}.csv"), index=False, encoding="utf-8-sig")


def gravar_excel():
    caminho = os.path.join(TAB_DIR, "tabelas_modelos.xlsx")
    try:
        with pd.ExcelWriter(caminho, engine="openpyxl") as xw:
            for aba, titulo, df, nota in TABELAS:
                df.to_excel(xw, sheet_name=aba[:31], index=False, startrow=2)
                ws = xw.sheets[aba[:31]]
                ws.cell(row=1, column=1, value=titulo)
                if nota:
                    ws.cell(row=len(df) + 5, column=1, value=f"Nota: {nota}")
                for col in ws.columns:
                    larg = max(len(str(c.value)) if c.value is not None else 0 for c in col[2:])
                    ws.column_dimensions[col[0].column_letter].width = min(max(10, larg + 2), 60)
        log(f"Tabelas salvas em {caminho}")
    except ImportError:
        log("openpyxl não instalado: tabelas salvas apenas em CSV.")


def estilo_graficos():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": COR_TEXTO2,
        "xtick.color": COR_TEXTO2, "ytick.color": COR_TEXTO2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6,
        "axes.titleweight": "bold", "axes.titlecolor": COR_TEXTO,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def salvar_figura(fig, nome, titulo):
    fig.savefig(os.path.join(FIG_DIR, f"{nome}.png"))
    plt.close(fig)
    log(f"  figura: {nome}.png — {titulo}")


NOTA_EF = ("Erros-padrão agrupados por corrida entre parênteses. *** p<0,01; ** p<0,05; "
           "* p<0,10. Todos os modelos incluem efeitos fixos de piloto, de circuito "
           "(separado por período nos traçados alterados) e de equipe × temporada.")
EF = "C(Driver) + C(CircuitoLayout) + C(TeamSeason)"


# =====================================================================
# 4.1 CARREGAR AS BASES E CRIAR AS VARIÁVEIS DOS MODELOS
# =====================================================================

def carregar():
    log("4.1 Carregando as bases finais...")
    pc = pd.read_csv(os.path.join(FINAL_DIR, "base_piloto_corrida.csv"), low_memory=False)
    voltas = pd.read_csv(os.path.join(FINAL_DIR, "base_voltas_ritmo.csv"), low_memory=False)

    for c in ["WetRace", "DNF", "DNF_Accident", "AnyDrivingPenalty", "Elegivel_H3"]:
        pc[c] = como_bool(pc[c])
    for c in ["AvgLapTimePct_vsCircuit_w", "AvgLapTimePct_vsRace_w", "ConsistencyResidStd_w",
              "LapTimePctStd_w", "TrackTemp_driver_mean", "TrackTemp_mean", "AirTemp_mean",
              "Humidity_mean", "WindSpeed_mean", "ShareRainLaps", "NumLaps", "NumSlowLaps",
              "NumDeletedLaps", "NumSpinsMsg", "NumPenalties_driving", "Season"]:
        pc[c] = a_numero(pc[c])
    for c in ["LapTimePct_vsCircuit_w", "LapTimePct_vsWindow", "TrackTemp", "Humidity",
              "WindSpeed", "TyreLife", "LapNumber", "Season"]:
        if c in voltas.columns:
            voltas[c] = a_numero(voltas[c])
    voltas["IsWetCompound"] = como_bool(voltas["IsWetCompound"])

    # Circuito × período do traçado (efeito fixo)
    for df in (pc, voltas):
        df["CircuitoLayout"] = [
            f"{loc}_{'novo' if s >= MUDANCAS_TRACADO[loc] else 'antigo'}"
            if loc in MUDANCAS_TRACADO else str(loc)
            for loc, s in zip(df["Location"], df["Season"])]

    # Variáveis dependentes de H2 (alternativa B: cada tipo de erro separado)
    pc["PerdaControle"] = ((pc["NumSpinsMsg"] > 0) | pc["DNF_Accident"]).astype(int)
    pc["PenalidadePilotagem"] = pc["AnyDrivingPenalty"].astype(int)
    ok = pc["NumLaps"] >= MIN_VOLTAS_TAXA
    pc["VoltasLentas_100"] = np.where(ok, 100 * pc["NumSlowLaps"] / pc["NumLaps"], np.nan)
    pc["VoltasAnuladas_100"] = np.where(ok, 100 * pc["NumDeletedLaps"] / pc["NumLaps"], np.nan)
    pc["Molhada"] = pc["WetRace"].astype(int)

    # Variáveis de H3 (experiência). Medida principal: log(1 + GPs disputados
    # antes da corrida), centrada na média. O log dá peso maior aos primeiros
    # GPs (aprender a pilotar na chuva acontece mais no início da carreira).
    # Com a variável centrada, o coeficiente de "Molhada" é o efeito da chuva
    # para um piloto de experiência média, e θ (Molhada × Experiência) diz
    # quanto esse efeito muda com a experiência.
    for c in ["GPsAnteriores", "Idade", "MolhadasAnteriores"]:
        pc[c] = a_numero(pc[c]) if c in pc.columns else np.nan
    if pc["GPsAnteriores"].notna().any():
        exp = np.log1p(pc["GPsAnteriores"])
        pc["Experiencia_c"] = exp - exp.mean()
        pc["Molhada_x_Exp"] = pc["Molhada"] * pc["Experiencia_c"]
        mediana = pc["GPsAnteriores"].median()
        pc["Veterano"] = np.where(pc["GPsAnteriores"].notna(),
                                  (pc["GPsAnteriores"] > mediana).astype(float), np.nan)
        pc["Molhada_x_Veterano"] = pc["Molhada"] * pc["Veterano"]
        em = np.log1p(pc["MolhadasAnteriores"])
        pc["ExpChuva_c"] = em - em.mean()
        pc["Molhada_x_ExpChuva"] = pc["Molhada"] * pc["ExpChuva_c"]
        pc["Idade_c"] = pc["Idade"] - pc["Idade"].mean()
        pc["Molhada_x_Idade"] = pc["Molhada"] * pc["Idade_c"]
        pc["PilotoMolhada"] = pc["Driver"].astype(str) + "_" + pc["Molhada"].astype(str)
        pc["EquipeMolhada"] = pc["Team"].astype(str) + "_" + pc["Molhada"].astype(str)
        log(f"  experiência: GPs anteriores de {pc['GPsAnteriores'].min():.0f} a "
            f"{pc['GPsAnteriores'].max():.0f} (mediana {mediana:.0f}); "
            f"{int(pc['GPsAnteriores'].notna().sum())} observações com histórico")
    else:
        log("  AVISO: sem GPsAnteriores — rode 01_coleta.py --historico e 02_tratamento.py. "
            "H3 (experiência) não será estimada.")

    # Inconsistência relativa ao grid no mesmo momento da corrida: desvio-
    # padrão do ritmo da volta em relação à mediana dos outros pilotos nas
    # voltas n±2. Retira a tendência comum (pista secando, combustível),
    # que inflaria a inconsistência em corridas molhadas sem ser do piloto.
    if "LapTimePct_vsWindow" in voltas.columns:
        inc = voltas.groupby(["RaceId", "Driver"])["LapTimePct_vsWindow"].std() \
            .rename("Inconsistencia_janela")
        pc = pc.merge(inc, left_on=["RaceId", "Driver"], right_index=True, how="left")
        lo, hi = pc["Inconsistencia_janela"].quantile([0.01, 0.99])
        pc["Inconsistencia_janela_w"] = pc["Inconsistencia_janela"].clip(lo, hi)

    # Temperatura centrada na média (o termo linear passa a ser o efeito na
    # temperatura média; reduz a correlação entre T e T²)
    seco = ~pc["WetRace"]
    t_media = pc.loc[seco, "TrackTemp_driver_mean"].mean()
    pc["TempPista_c"] = pc["TrackTemp_driver_mean"] - t_media
    pc["TempPista_c2"] = pc["TempPista_c"] ** 2
    pc["TempPistaCorrida_c"] = pc["TrackTemp_mean"] - pc["TrackTemp_mean"].mean()
    pc["TempAr_c"] = pc["AirTemp_mean"] - pc["AirTemp_mean"].mean()
    pc["TempAr_c2"] = pc["TempAr_c"] ** 2
    voltas["TempPista_c"] = voltas["TrackTemp"] - t_media
    voltas["TempPista_c2"] = voltas["TempPista_c"] ** 2
    log(f"  temperatura da pista centrada em {t_media:.1f} °C (média das corridas secas)")
    log(f"  {len(pc)} observações piloto-corrida; {pc['RaceId'].nunique()} corridas; "
        f"{int(pc.groupby('RaceId')['WetRace'].first().sum())} molhadas")
    return pc, voltas, t_media


# =====================================================================
# 4.2 H1 — TEMPERATURA x RITMO E CONSISTÊNCIA (corridas secas)
# =====================================================================
# Modelo: Y_ij = β1·T_ij + β2·T²_ij + γ·Controles_j + α_piloto + α_circuito
#               + α_equipe×temporada + ε_ij
# H1 prevê β1 > 0 (ritmo mais lento e mais irregular com pista mais quente).
# β2 capta a não linearidade (janela de operação dos pneus).

ROT_H1 = {"TempPista_c": "Temperatura da pista (°C, centrada)",
          "TempPista_c2": "Temperatura da pista² ",
          "Humidity_mean": "Umidade relativa (%)",
          "WindSpeed_mean": "Velocidade do vento (m/s)",
          "TyreLife": "Idade do pneu (voltas)",
          "LapNumber": "Número da volta"}


def hipotese_h1(pc):
    log("4.2 H1 — temperatura x ritmo e consistência (corridas secas)...")
    seco = pc[~pc["WetRace"]].copy()
    ctrl = "Humidity_mean + WindSpeed_mean"
    m = {
        "(1) Ritmo — linear": mqo_cluster(
            f"AvgLapTimePct_vsCircuit_w ~ TempPista_c + {ctrl} + {EF}", seco),
        "(2) Ritmo — quadrático": mqo_cluster(
            f"AvgLapTimePct_vsCircuit_w ~ TempPista_c + TempPista_c2 + {ctrl} + {EF}", seco),
        "(3) Inconsistência (stint)": mqo_cluster(
            f"ConsistencyResidStd_w ~ TempPista_c + TempPista_c2 + {ctrl} + {EF}", seco),
        "(4) Inconsistência (janela)": mqo_cluster(
            f"Inconsistencia_janela_w ~ TempPista_c + TempPista_c2 + {ctrl} + {EF}", seco),
    }
    tab = tabela_regressao(m, ["TempPista_c", "TempPista_c2", "Humidity_mean", "WindSpeed_mean"],
                           ROT_H1, EXTRAS_PADRAO)
    registrar_tabela("M1_H1_temperatura", "Tabela M1 — Temperatura da pista, ritmo e consistência (H1)",
                     tab, "Corridas secas, nível piloto-corrida. Variáveis dependentes: (1)–(2) ritmo "
                          "médio em % acima da referência seca do circuito; (3) desvio-padrão do resíduo "
                          "do ritmo no stint (p.p.); (4) desvio-padrão do ritmo em relação ao grid nas "
                          "mesmas voltas (p.p.). Temperatura centrada na média das corridas secas. " + NOTA_EF)
    b1, ep1, p1 = coef(m["(1) Ritmo — linear"], "TempPista_c")
    log(f"  H1 (1): +1 °C na pista -> ritmo {b1:+.4f} p.p. (EP {ep1:.4f}, p = {fmt_p(p1)}); "
        f"+10 °C -> {10 * b1:+.3f} p.p.")
    for nome in ["(3) Inconsistência (stint)", "(4) Inconsistência (janela)"]:
        b, ep, p = coef(m[nome], "TempPista_c")
        b2, _, p2 = coef(m[nome], "TempPista_c2")
        log(f"  H1 {nome}: linear {b:+.4f} (p = {fmt_p(p)}), quadrático {b2:+.5f} (p = {fmt_p(p2)})")
    return m


# =====================================================================
# 4.3 H1 — MODELO NO NÍVEL DA VOLTA (robustez)
# =====================================================================

def hipotese_h1_voltas(voltas):
    log("4.3 H1 — modelo no nível da volta (robustez)...")
    v = voltas[~voltas["IsWetCompound"]].dropna(
        subset=["LapTimePct_vsCircuit_w", "TempPista_c", "TyreLife", "LapNumber"]).copy()
    v["Compound"] = v["Compound"].astype(str)
    f = (f"LapTimePct_vsCircuit_w ~ TempPista_c + TempPista_c2 + TyreLife + LapNumber "
         f"+ C(Compound) + {EF}")
    m = {"(5) Ritmo da volta": mqo_cluster(f, v)}
    tab = tabela_regressao(m, ["TempPista_c", "TempPista_c2", "TyreLife", "LapNumber"],
                           ROT_H1, EXTRAS_PADRAO)
    registrar_tabela("M2_H1_voltas", "Tabela M2 — Temperatura da pista e ritmo, nível da volta (H1, robustez)",
                     tab, "Voltas de ritmo com pneu de pista seca. Controles: idade do pneu, número da "
                          "volta (consumo de combustível) e composto. " + NOTA_EF)
    b, ep, p = coef(m["(5) Ritmo da volta"], "TempPista_c")
    log(f"  H1 (volta): +1 °C -> {b:+.4f} p.p. (p = {fmt_p(p)}); N = {int(m['(5) Ritmo da volta'].nobs)}")
    return m


# =====================================================================
# 4.4 H2 — CHUVA x CADA TIPO DE ERRO (alternativa B)
# =====================================================================
# Modelo: Y_ij = δ·Molhada_j + β·T_j + α_piloto + α_circuito + α_equipe×temporada + ε_ij
# δ = diferença média em Y entre corrida molhada e seca, para o mesmo piloto,
# no mesmo circuito e com o mesmo carro. Cada tipo de erro em uma coluna.

DEP_H2 = {
    "PerdaControle": "(1) Rodada ou abandono por acidente (0/1)",
    "VoltasLentas_100": "(2) Voltas lentas suspeitas por 100 voltas",
    "VoltasAnuladas_100": "(3) Voltas anuladas por 100 voltas",
    "PenalidadePilotagem": "(4) Penalidade de pilotagem (0/1)",
    "Inconsistencia_janela_w": "(5) Inconsistência vs. grid (p.p.)",
}


def hipotese_h2(pc):
    log("4.4 H2 — chuva x tipos de erro...")
    m = {rotulo: mqo_cluster(f"{y} ~ Molhada + TempPistaCorrida_c + {EF}", pc)
         for y, rotulo in DEP_H2.items()}
    tab = tabela_regressao(m, ["Molhada", "TempPistaCorrida_c"],
                           {"Molhada": "Corrida molhada (0/1)",
                            "TempPistaCorrida_c": "Temperatura da pista (°C, centrada)"},
                           EXTRAS_PADRAO + [("Média da variável nas corridas secas",
                                             lambda r: f"{r.dados.loc[r.dados['Molhada'] == 0, r.y_nome].mean():.4f}")])
    registrar_tabela("M3_H2_chuva_erros", "Tabela M3 — Corrida molhada e tipos de erro (H2)", tab,
                     "Nível piloto-corrida, todas as corridas da amostra principal. (1) e (4): modelo de "
                     "probabilidade linear (coeficiente = variação em pontos de probabilidade). (2) e (3): "
                     f"somente pilotos com >= {MIN_VOLTAS_TAXA} voltas. Abandono por acidente: causa "
                     "oficial em 2021–2022 e aproximação validada nas demais. Safety Car e pneu de chuva "
                     "não entram como controles por serem consequências da chuva. " + NOTA_EF)
    efeitos = []
    for y, rotulo in DEP_H2.items():
        b, ep, p = coef(m[rotulo], "Molhada")
        res = m[rotulo]
        base = res.dados.loc[res.dados["Molhada"] == 0, res.y_nome].mean()
        efeitos.append({"Erro": rotulo, "b": b, "ep": ep, "p": p, "media_seca": base})
        log(f"  H2 {rotulo}: molhada {b:+.4f} (EP {ep:.4f}, p = {fmt_p(p)}); "
            f"média no seco = {base:.4f}")
    return m, pd.DataFrame(efeitos)


# =====================================================================
# 4.5 H3 — EXPERIÊNCIA DO PILOTO MODERA O EFEITO DA CHUVA
# =====================================================================
# Modelo: Y_ij = δ·Molhada_j + θ·(Molhada_j × Exp_ij) + φ·Exp_ij + β·T_j
#               + α_piloto + α_circuito + α_equipe×temporada + ε_ij
# Exp_ij = log(1 + GPs disputados pelo piloto i antes da corrida j), centrada.
# δ = efeito da chuva para um piloto de experiência média.
# θ = variação desse efeito quando a experiência aumenta (1 unidade de log
#     ≈ multiplicar os GPs anteriores por 2,7). H3 prevê θ < 0 (para erros e
#     ritmo, valores maiores são piores): pilotos mais experientes sofrem
#     MENOS com a chuva. Teste unilateral de H0: θ >= 0.
# O efeito fixo de piloto retira o nível de cada piloto; a experiência varia
# entre pilotos e, para o mesmo piloto, ao longo da carreira.

DEP_H3 = {
    "PerdaControle": "(1) Rodada ou acidente (0/1)",
    "VoltasLentas_100": "(2) Voltas lentas por 100 voltas",
    "Inconsistencia_janela_w": "(3) Inconsistência vs. grid (p.p.)",
    "AvgLapTimePct_vsRace_w": "(4) Ritmo relativo ao grid (%)",
}
ROT_H3 = {"Molhada": "Corrida molhada (0/1) — piloto de experiência média",
          "Molhada_x_Exp": "Molhada × Experiência (θ)",
          "Experiencia_c": "Experiência: log(1 + GPs anteriores), centrada",
          "TempPistaCorrida_c": "Temperatura da pista (°C, centrada)"}


def efeito_chuva_em(res, valor, termo="Molhada_x_Exp"):
    """Efeito da chuva para um piloto com Experiência = valor: δ + θ·valor,
    com erro-padrão pela covariância agrupada (combinação linear)."""
    if "Molhada" not in res.params.index or termo not in res.params.index:
        return np.nan, np.nan, np.nan
    V = res.cov_params()
    b = res.params["Molhada"] + valor * res.params[termo]
    ep = np.sqrt(V.loc["Molhada", "Molhada"] + valor ** 2 * V.loc[termo, termo]
                 + 2 * valor * V.loc["Molhada", termo])
    p = 2 * stats.t.sf(abs(b / ep), res.gl)
    return b, ep, p


def p_unilateral(res, termo="Molhada_x_Exp"):
    """p-valor do teste unilateral H0: θ >= 0 contra H1: θ < 0."""
    if termo not in res.params.index:
        return np.nan
    return float(stats.t.cdf(res.params[termo] / res.bse[termo], res.gl))


def hipotese_h3(pc):
    log("4.5 H3 — experiência do piloto x efeito da chuva...")
    if "Molhada_x_Exp" not in pc.columns:
        log("  H3 não estimada (sem dados de experiência).")
        return None
    d = pc.dropna(subset=["Experiencia_c"]).copy()
    exp_media = np.log1p(pc["GPsAnteriores"]).mean()
    q = d["GPsAnteriores"].quantile(list(PERCENTIS_EXPERIENCIA))
    gps_novato, gps_vet = q.iloc[0], q.iloc[1]
    x_novato, x_vet = np.log1p(gps_novato) - exp_media, np.log1p(gps_vet) - exp_media

    m = {rot: mqo_cluster(f"{y} ~ Molhada + Molhada_x_Exp + Experiencia_c + TempPistaCorrida_c + {EF}", d)
         for y, rot in DEP_H3.items()}

    def fmt_efeito(res, x):
        b, ep, p = efeito_chuva_em(res, x)
        return "" if pd.isna(b) else f"{b:.4f}{estrelas(p)} ({ep:.4f})"

    extras = [
        ("p-valor unilateral (H0: θ ≥ 0)", lambda r: fmt_p(p_unilateral(r))),
        (f"Efeito da chuva — menos experiente (p{int(100 * PERCENTIS_EXPERIENCIA[0])}: "
         f"{gps_novato:.0f} GPs)", lambda r: fmt_efeito(r, x_novato)),
        (f"Efeito da chuva — mais experiente (p{int(100 * PERCENTIS_EXPERIENCIA[1])}: "
         f"{gps_vet:.0f} GPs)", lambda r: fmt_efeito(r, x_vet)),
        ("Média da variável nas corridas secas",
         lambda r: f"{r.dados.loc[r.dados['Molhada'] == 0, r.y_nome].mean():.4f}"),
    ] + EXTRAS_PADRAO
    tab = tabela_regressao(m, ["Molhada", "Molhada_x_Exp", "Experiencia_c", "TempPistaCorrida_c"],
                           ROT_H3, extras)
    registrar_tabela("M4_H3_experiencia", "Tabela M4 — Experiência do piloto e efeito da chuva (H3)", tab,
                     "Nível piloto-corrida. Experiência = log(1 + nº de GPs em que o piloto largou antes da "
                     "corrida, carreira inteira), centrada na média; o coeficiente de corrida molhada é o "
                     "efeito da chuva para um piloto de experiência média e θ mostra como esse efeito varia "
                     "com a experiência (H3: θ < 0). Efeito da chuva nos percentis de experiência = δ + θ·Exp, "
                     "com erro-padrão entre parênteses. (1): modelo de probabilidade linear. (2): pilotos com "
                     f">= {MIN_VOLTAS_TAXA} voltas. " + NOTA_EF)
    resumo = []
    for y, rot in DEP_H3.items():
        res = m[rot]
        b, ep, p = coef(res, "Molhada_x_Exp")
        bn, epn, _ = efeito_chuva_em(res, x_novato)
        bv, epv, _ = efeito_chuva_em(res, x_vet)
        resumo.append({"Y": y, "rotulo": rot, "theta": b, "ep": ep, "p": p, "p_uni": p_unilateral(res),
                       "novato": bn, "ep_novato": epn, "veterano": bv, "ep_veterano": epv})
        log(f"  H3 {rot}: θ = {b:+.4f} (EP {ep:.4f}; p bilateral = {fmt_p(p)}; "
            f"p unilateral = {fmt_p(p_unilateral(res))}); chuva p/ {gps_novato:.0f} GPs = {bn:+.4f}, "
            f"p/ {gps_vet:.0f} GPs = {bv:+.4f}")
    info = {"modelos": m, "exp_media": exp_media, "resumo": pd.DataFrame(resumo), "dados": d}
    robustez_h3(d)
    return info


def robustez_h3(d):
    """Tabela M4b: θ em especificações alternativas (colunas = variáveis
    dependentes, linhas = especificações)."""
    log("4.5b H3 — robustez...")
    base = "TempPistaCorrida_c + "
    especs = [
        ("Principal (Tabela M4)",
         "Molhada + Molhada_x_Exp + Experiencia_c", EF, d, "RaceId", "Molhada_x_Exp"),
        ("Experiência em dummy: veterano (acima da mediana de GPs)",
         "Molhada + Molhada_x_Veterano + Veterano", EF, d, "RaceId", "Molhada_x_Veterano"),
        ("Experiência na chuva: log(1 + corridas molhadas anteriores, desde 2021)",
         "Molhada + Molhada_x_ExpChuva + ExpChuva_c", EF, d, "RaceId", "Molhada_x_ExpChuva"),
        ("Controle de idade: + Molhada × Idade",
         "Molhada + Molhada_x_Exp + Experiencia_c + Molhada_x_Idade + Idade_c", EF, d, "RaceId",
         "Molhada_x_Exp"),
        ("Efeito fixo piloto × chuva (só variação dentro do piloto)",
         "Molhada + Molhada_x_Exp + Experiencia_c", EF + " + C(PilotoMolhada)", d, "RaceId",
         "Molhada_x_Exp"),
        ("Efeito fixo equipe × chuva (carro na chuva)",
         "Molhada + Molhada_x_Exp + Experiencia_c", EF + " + C(EquipeMolhada)", d, "RaceId",
         "Molhada_x_Exp"),
        ("Erros agrupados por corrida e por piloto (agrupamento duplo)",
         "Molhada + Molhada_x_Exp + Experiencia_c", EF, d, ["RaceId", "Driver"], "Molhada_x_Exp"),
        ("Sem a temporada 2026",
         "Molhada + Molhada_x_Exp + Experiencia_c", EF, d[d["Season"] < 2026], "RaceId", "Molhada_x_Exp"),
    ]
    linhas = []
    for desc, rhs, efs, dados, cl, termo in especs:
        l1, l2 = {"Especificação": desc, "Coeficiente de interesse": termo.replace("_x_", " × ")},                  {"Especificação": "", "Coeficiente de interesse": ""}
        for y, rot in DEP_H3.items():
            try:
                res = mqo_cluster(f"{y} ~ {rhs} + {base}{efs}", dados, cluster=cl)
                b, ep, p = coef(res, termo)
            except Exception as e:      # noqa: BLE001 — registra e segue
                log(f"  H3 robustez '{desc}' / {y}: não estimado ({e})")
                b = ep = p = np.nan
            l1[rot] = "" if pd.isna(b) else f"{b:.4f}{estrelas(p)}"
            l2[rot] = "" if pd.isna(ep) else f"({ep:.4f})"
            if y in ("PerdaControle", "VoltasLentas_100") and not pd.isna(b):
                log(f"  H3 robustez — {desc} — {y}: {b:+.4f} (p = {fmt_p(p)})")
        linhas += [l1, l2]
    registrar_tabela("M4b_H3_robustez", "Tabela M4b — Robustez de H3 (experiência × chuva)",
                     pd.DataFrame(linhas),
                     "Cada célula: coeficiente da interação entre corrida molhada e a medida de experiência, "
                     "erro-padrão entre parênteses. Na linha 'piloto × chuva', o efeito médio da chuva de cada "
                     "piloto é absorvido, e θ vem apenas da mudança de experiência do mesmo piloto ao longo "
                     "das temporadas. Agrupamento duplo: inferência com t de (min(corridas, pilotos) − 1) graus "
                     "de liberdade. " + NOTA_EF)


# =====================================================================
# 4.5c H3 — ANÁLISE COMPLEMENTAR: DIFERENÇAS PILOTO A PILOTO
# =====================================================================
# Pergunta mais ampla (a do 1º bimestre): o efeito da chuva difere entre os
# pilotos, seja qual for o motivo? Mantida como análise complementar.
# Modelo: Y_ij = Σ_i δ_i·(Piloto_i × Molhada_j) + β·T_j + efeitos fixos + ε_ij
# Cada piloto tem seu próprio efeito da chuva δ_i. H3 é testada pelo teste
# de Wald (F) conjunto de H0: δ_1 = δ_2 = ... = δ_k (todos iguais), usando
# a matriz de covariância agrupada por corrida.

def efeitos_por_piloto(res, termo):
    """Extrai δ_i de cada piloto e testa a igualdade de todos (Wald/F)."""
    nomes = [n for n in res.params.index if n.startswith("C(Driver)[") and n.endswith(f"]:{termo}")]
    linhas = []
    for n in nomes:
        piloto = n.split("[")[1].split("]")[0]
        linhas.append({"Piloto": piloto, "Efeito": res.params[n], "EP": res.bse[n],
                       "IC95% inf.": res.conf_int().loc[n, 0], "IC95% sup.": res.conf_int().loc[n, 1],
                       "p-valor": res.pvalues[n]})
    idx = [list(res.params.index).index(n) for n in nomes]
    R = np.zeros((len(idx) - 1, len(res.params)))
    for k in range(1, len(idx)):
        R[k - 1, idx[k]], R[k - 1, idx[0]] = 1, -1
    teste = res.f_test(R)
    return pd.DataFrame(linhas), float(np.squeeze(teste.fvalue)), float(teste.pvalue), int(teste.df_num)


def h3_complementar(pc):
    log("4.5c H3 complementar — heterogeneidade entre pilotos...")
    d = pc[pc["Elegivel_H3"]].copy()
    # O termo C(Driver) entra nos efeitos fixos; C(Driver):Molhada dá um
    # efeito da chuva para cada piloto (sem termo "Molhada" isolado).
    ef_sem_piloto = "C(CircuitoLayout) + C(TeamSeason)"
    testes, tabelas = [], {}
    especificacoes = [
        ("Ritmo relativo ao grid (%)", "AvgLapTimePct_vsRace_w", "Molhada", d),
        ("Rodada ou abandono por acidente (0/1)", "PerdaControle", "Molhada", d),
        ("Voltas lentas por 100 voltas", "VoltasLentas_100", "Molhada", d),
        ("Inconsistência vs. grid (p.p.)", "Inconsistencia_janela_w", "Molhada", d),
    ]
    for rotulo, y, termo, dados in especificacoes:
        res = mqo_cluster(f"{y} ~ C(Driver) + C(Driver):{termo} + TempPistaCorrida_c + "
                          f"{ef_sem_piloto}", dados)
        tab, F, p, gl = efeitos_por_piloto(res, termo)
        tabelas[rotulo] = tab
        testes.append({"Variável dependente": rotulo, "Interação": "Piloto × corrida molhada",
                       "Pilotos": len(tab), "F": F, "gl": gl, "p-valor": fmt_p(p), "Sig.": estrelas(p),
                       "Observações": int(res.nobs), "Corridas (clusters)": res.n_clusters})
        log(f"  H3 compl. {rotulo} × chuva: F({gl}) = {F:.2f}, p = {fmt_p(p)}")

    # Piloto × temperatura (corridas secas)
    seco = pc[~pc["WetRace"] & pc["Elegivel_H3"]].copy()
    res = mqo_cluster(f"AvgLapTimePct_vsRace_w ~ C(Driver) + C(Driver):TempPistaCorrida_c + "
                      f"Humidity_mean + WindSpeed_mean + {ef_sem_piloto}", seco)
    tab_t, F, p, gl = efeitos_por_piloto(res, "TempPistaCorrida_c")
    tabelas["Sensibilidade ao calor"] = tab_t
    testes.append({"Variável dependente": "Ritmo relativo ao grid (%) — corridas secas",
                   "Interação": "Piloto × temperatura da pista", "Pilotos": len(tab_t), "F": F,
                   "gl": gl, "p-valor": fmt_p(p), "Sig.": estrelas(p),
                   "Observações": int(res.nobs), "Corridas (clusters)": res.n_clusters})
    log(f"  H3 compl. ritmo × temperatura: F({gl}) = {F:.2f}, p = {fmt_p(p)}")

    # --- Robustez de H3 ---
    # (a) Controle do desempenho do CARRO na chuva: efeito fixo equipe ×
    #     corrida molhada. A diferença entre pilotos passa a vir sobretudo da
    #     comparação com o companheiro de equipe (mesmo carro).
    # (b) Só pilotos com mais corridas molhadas (menos restrições no teste F,
    #     que tende a rejeitar demais quando há muitas restrições).
    d["EquipeMolhada"] = d["Team"].astype(str) + "_" + d["Molhada"].astype(str)
    d8 = d[d["Driver"].isin(d.loc[d["Molhada"] == 1].groupby("Driver")["RaceId"].nunique()
                            .loc[lambda x: x >= MIN_MOLHADAS_H3_ROBUSTEZ].index)]
    variantes = [
        ("com efeito fixo equipe × chuva (carro na chuva)", d, " + C(EquipeMolhada)"),
        (f"só pilotos com >= {MIN_MOLHADAS_H3_ROBUSTEZ} corridas molhadas", d8, ""),
    ]
    for desc, dados, extra in variantes:
        if dados["Driver"].nunique() < 3:
            log(f"  H3 compl. — {desc}: pilotos insuficientes, variante não estimada")
            continue
        for rotulo, y, termo, _ in especificacoes:
            res = mqo_cluster(f"{y} ~ C(Driver) + C(Driver):{termo} + TempPistaCorrida_c + "
                              f"{ef_sem_piloto}{extra}", dados)
            tab, F, p, gl = efeitos_por_piloto(res, termo)
            testes.append({"Variável dependente": rotulo,
                           "Interação": f"Piloto × corrida molhada — {desc}",
                           "Pilotos": len(tab), "F": F, "gl": gl, "p-valor": fmt_p(p),
                           "Sig.": estrelas(p), "Observações": int(res.nobs),
                           "Corridas (clusters)": res.n_clusters})
            log(f"  H3 compl. {rotulo} — {desc}: F({gl}) = {F:.2f}, p = {fmt_p(p)}")

    # (c) Teste de permutação (inferência por aleatorização): sorteia quais
    #     corridas são "molhadas" (mantendo o total) e recalcula o F. O p-valor
    #     é a proporção de sorteios com F >= F observado. Não depende da
    #     aproximação do teste F com erros agrupados.
    rng = np.random.default_rng(SEMENTE)
    corridas = d.groupby("RaceId")["Molhada"].first()
    for rotulo, y, termo, _ in especificacoes[:2]:
        f_obs = next(t["F"] for t in testes if t["Variável dependente"] == rotulo)
        f_perm = []
        for _ in range(N_PERMUTACOES):
            sorteio = pd.Series(rng.permutation(corridas.values), index=corridas.index)
            dp = d.assign(Molhada=d["RaceId"].map(sorteio))
            res = mqo_cluster(f"{y} ~ C(Driver) + C(Driver):{termo} + TempPistaCorrida_c + "
                              f"{ef_sem_piloto}", dp)
            f_perm.append(efeitos_por_piloto(res, termo)[1])
        p_perm = (1 + np.sum(np.array(f_perm) >= f_obs)) / (1 + N_PERMUTACOES)
        testes.append({"Variável dependente": rotulo,
                       "Interação": f"Piloto × corrida molhada — teste de permutação ({N_PERMUTACOES} sorteios)",
                       "Pilotos": np.nan, "F": f_obs, "gl": np.nan, "p-valor": fmt_p(p_perm),
                       "Sig.": estrelas(p_perm), "Observações": np.nan, "Corridas (clusters)": np.nan})
        log(f"  H3 compl. {rotulo} — permutação: p = {fmt_p(p_perm)} ({N_PERMUTACOES} sorteios)")

    t4 = pd.DataFrame(testes)
    t4["F"] = t4["F"].round(3)
    registrar_tabela("M4c_H3_compl_testes", "Tabela M4c — Testes de heterogeneidade entre pilotos "
                     "(H3, análise complementar)", t4,
                     "Teste de Wald (F) de H0: o efeito da chuva (ou da temperatura) é igual para todos "
                     "os pilotos, com covariância agrupada por corrida. Com muitas restrições, esse teste "
                     "tende a rejeitar H0 com frequência maior que a nominal; por isso são reportados o "
                     "controle do carro na chuva, a amostra de pilotos com mais corridas molhadas e o "
                     "teste de permutação. Pilotos elegíveis: >= 4 corridas "
                     "molhadas e >= 10 secas. Modelos com efeitos fixos de piloto, circuito e equipe × "
                     "temporada e temperatura da pista como controle. *** p<0,01; ** p<0,05; * p<0,10.")
    t_ritmo = tabelas["Ritmo relativo ao grid (%)"].sort_values("Efeito").round(4)
    t_ritmo["p-valor"] = t_ritmo["p-valor"].map(fmt_p)
    registrar_tabela("M4d_H3_compl_pilotos", "Tabela M4d — Efeito da chuva no ritmo relativo, por piloto "
                     "(H3, análise complementar)",
                     t_ritmo, "Efeito = diferença no ritmo relativo ao grid entre corridas molhadas e secas "
                              "(p.p.; negativo = ganha do grid na chuva), com efeitos fixos e erros agrupados "
                              "por corrida. Pilotos com poucas corridas molhadas têm intervalos largos.")
    return tabelas


# =====================================================================
# 4.6 ROBUSTEZ DE H1 E H2
# =====================================================================

def volta_lenta_limite(limite):
    """Recalcula as voltas lentas por 100 voltas com outro limite (%)."""
    v = pd.read_csv(os.path.join(TRATADOS_DIR, "voltas_tratadas.csv"),
                    usecols=["RaceId", "Driver", "IsCleanLap", "LapTimePct_vsWindow", "AmostraPrincipal"],
                    low_memory=False)
    v = v[como_bool(v["AmostraPrincipal"]) & como_bool(v["IsCleanLap"])]
    v["lenta"] = a_numero(v["LapTimePct_vsWindow"]) > limite
    return v.groupby(["RaceId", "Driver"])["lenta"].sum().rename(f"lentas_{limite:g}")


def robustez(pc):
    log("4.6 Robustez...")
    seco = pc[~pc["WetRace"]]
    linhas = []

    def add(hip, descricao, res, var):
        b, ep, p = coef(res, var)
        linhas.append({"Hipótese": hip, "Especificação": descricao,
                       "Coeficiente": f"{b:.4f}{estrelas(p)}", "Erro-padrão": f"({ep:.4f})",
                       "p-valor": fmt_p(p), "Observações": int(res.nobs)})

    ctrl = "Humidity_mean + WindSpeed_mean"
    # H1
    add("H1", "Principal: ritmo, temperatura da pista (linear)",
        mqo_cluster(f"AvgLapTimePct_vsCircuit_w ~ TempPista_c + {ctrl} + {EF}", seco), "TempPista_c")
    add("H1", "Temperatura do ar no lugar da pista",
        mqo_cluster(f"AvgLapTimePct_vsCircuit_w ~ TempAr_c + {ctrl} + {EF}", seco), "TempAr_c")
    add("H1", "Temperatura média da corrida (não do piloto)",
        mqo_cluster(f"AvgLapTimePct_vsCircuit_w ~ TempPistaCorrida_c + {ctrl} + {EF}", seco),
        "TempPistaCorrida_c")
    add("H1", "Ritmo relativo ao grid como variável dependente",
        mqo_cluster(f"AvgLapTimePct_vsRace_w ~ TempPista_c + {ctrl} + {EF}", seco), "TempPista_c")
    add("H1", "Sem a temporada 2026 (novo regulamento)",
        mqo_cluster(f"AvgLapTimePct_vsCircuit_w ~ TempPista_c + {ctrl} + {EF}",
                    seco[seco["Season"] < 2026]), "TempPista_c")
    add("H1", "Sem efeito fixo de piloto",
        mqo_cluster(f"AvgLapTimePct_vsCircuit_w ~ TempPista_c + {ctrl} + C(CircuitoLayout) + "
                    f"C(TeamSeason)", seco), "TempPista_c")

    # H2 — rodada/acidente e voltas lentas
    f_pc = f"PerdaControle ~ Molhada + TempPistaCorrida_c + {EF}"
    add("H2", "Principal: rodada ou acidente (MPL)", mqo_cluster(f_pc, pc), "Molhada")
    r22 = mqo_cluster(f_pc, pc[pc["Season"] <= 2022])
    if "Molhada" in r22.params.index:
        add("H2", "Rodada ou acidente — só 2021–2022 (causa oficial)", r22, "Molhada")
    else:
        log("  2021–2022: chuva não varia dentro dos circuitos nessa subamostra; "
            "estimado sem efeito fixo de circuito")
    add("H2", "Rodada ou acidente — só 2021–2022 (causa oficial), sem efeito fixo de circuito",
        mqo_cluster("PerdaControle ~ Molhada + TempPistaCorrida_c + C(Driver) + C(TeamSeason)",
                    pc[pc["Season"] <= 2022]), "Molhada")
    add("H2", "Rodada ou acidente — chuva medida pela proporção de voltas do piloto",
        mqo_cluster(f"PerdaControle ~ ShareRainLaps + TempPistaCorrida_c + {EF}", pc), "ShareRainLaps")
    add("H2", "Rodada ou acidente — sem 2026", mqo_cluster(f_pc, pc[pc["Season"] < 2026]), "Molhada")
    add("H2", "Voltas lentas — Poisson (contagem, exposição = voltas)",
        poisson_cluster(f"NumSlowLaps ~ Molhada + TempPistaCorrida_c + {EF}", pc, "NumLaps"), "Molhada")
    add("H2", "Voltas anuladas — Poisson (contagem, exposição = voltas)",
        poisson_cluster(f"NumDeletedLaps ~ Molhada + TempPistaCorrida_c + {EF}", pc, "NumLaps"), "Molhada")
    try:
        for lim in LIMITES_VOLTA_LENTA_ROBUSTEZ:
            s = volta_lenta_limite(lim)
            d = pc.merge(s, left_on=["RaceId", "Driver"], right_index=True, how="left")
            d["y"] = np.where(d["NumLaps"] >= MIN_VOLTAS_TAXA, 100 * d[s.name] / d["NumLaps"], np.nan)
            add("H2", f"Voltas lentas por 100 voltas — limite de {lim:g}% (principal: 10%)",
                mqo_cluster(f"y ~ Molhada + TempPistaCorrida_c + {EF}", d), "Molhada")
    except (FileNotFoundError, ValueError, KeyError) as e:
        log(f"  robustez do limite de volta lenta não executada ({e})")

    t5 = pd.DataFrame(linhas)
    registrar_tabela("M5_robustez", "Tabela M5 — Testes de robustez (H1 e H2)", t5,
                     "Cada linha é um modelo. H1: coeficiente da temperatura (°C) no ritmo, corridas secas. "
                     "H2: coeficiente de corrida molhada (ou da proporção de voltas com chuva). Poisson: "
                     "coeficiente em log (exp(b) − 1 = variação percentual). " + NOTA_EF)
    return t5


# =====================================================================
# 4.7 FIGURAS DOS MODELOS
# =====================================================================

def figuras(m_h1, efeitos_h2, tab_h3, info_h3, t_media, pc):
    log("4.7 Figuras dos modelos...")
    # Figura 10: efeito previsto da temperatura (modelo quadrático) com IC 95%
    res = m_h1["(2) Ritmo — quadrático"]
    seco = pc[~pc["WetRace"]]
    tc = np.linspace(seco["TempPista_c"].quantile(.02), seco["TempPista_c"].quantile(.98), 60)
    nomes = list(res.params.index)
    i1, i2 = nomes.index("TempPista_c"), nomes.index("TempPista_c2")
    V = res.cov_params().values
    efeito = res.params.iloc[i1] * tc + res.params.iloc[i2] * tc ** 2
    var = tc ** 2 * V[i1, i1] + tc ** 4 * V[i2, i2] + 2 * tc ** 3 * V[i1, i2]
    ic = 1.96 * np.sqrt(var)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(tc + t_media, efeito - ic, efeito + ic, color=COR_SECO, alpha=0.18, linewidth=0)
    ax.plot(tc + t_media, efeito, color=COR_SECO)
    ax.axhline(0, color=COR_NEUTRA, linewidth=1)
    ax.set_xlabel("Temperatura da pista (°C)")
    ax.set_ylabel("Ritmo em relação à temperatura média (p.p.)")
    ax.set_title("Figura 10 — Efeito estimado da temperatura no ritmo (H1)")
    fig.text(0.01, -0.04, f"Modelo (2) da Tabela M1; faixa sombreada: IC 95% (erros agrupados por corrida). "
             f"Zero = temperatura média ({t_media:.1f} °C).", fontsize=8, color=COR_TEXTO2)
    salvar_figura(fig, "F10_H1_efeito_temperatura", "efeito previsto da temperatura")

    # Figura 11: efeito da chuva em cada tipo de erro, em % da média no seco
    e = efeitos_h2.copy()
    e["rel"] = 100 * e["b"] / e["media_seca"]
    e["ic"] = 100 * 1.96 * e["ep"] / e["media_seca"].abs()
    e = e.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(e))
    ax.axvline(0, color=COR_NEUTRA, linewidth=1)
    ax.hlines(y, e["rel"] - e["ic"], e["rel"] + e["ic"], color=COR_MOLHADO, linewidth=1.5, alpha=0.5)
    ax.plot(e["rel"], y, "o", color=COR_MOLHADO, markersize=7)
    ax.set_yticks(y, [t.split(") ", 1)[1] for t in e["Erro"]])
    ax.set_xlabel("Diferença na corrida molhada, em % da média nas corridas secas")
    ax.set_title("Figura 11 — Efeito estimado da chuva em cada tipo de erro (H2)")
    fig.text(0.01, -0.06, "Coeficientes da Tabela M3 divididos pela média no seco; IC 95% "
             "(erros agrupados por corrida).", fontsize=8, color=COR_TEXTO2)
    salvar_figura(fig, "F11_H2_efeito_chuva", "efeito da chuva por tipo de erro")

    # Figura 12: efeito da chuva conforme a experiência (H3), com IC 95%
    if info_h3 is not None:
        d = info_h3["dados"]
        gps = np.unique(np.round(np.geomspace(1, d["GPsAnteriores"].quantile(.98) + 1, 60) - 1))
        x = np.log1p(gps) - info_h3["exp_media"]
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (rot, res) in zip(axs.ravel(), info_h3["modelos"].items()):
            efeitos = np.array([efeito_chuva_em(res, xi)[:2] for xi in x])
            if np.isnan(efeitos).all():
                ax.set_visible(False)
                continue
            b, ep = efeitos[:, 0], efeitos[:, 1]
            tc = stats.t.ppf(0.975, res.gl)
            ax.fill_between(gps, b - tc * ep, b + tc * ep, color=COR_MOLHADO, alpha=0.18, linewidth=0)
            ax.plot(gps, b, color=COR_MOLHADO)
            ax.axhline(0, color=COR_NEUTRA, linewidth=1)
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(0, gps.max())
            ax.set_title(rot.split(") ", 1)[1], fontsize=10)
            ax.set_xlabel("GPs disputados antes da corrida")
            ax.set_ylabel("Efeito da corrida molhada")
        fig.suptitle("Figura 12 — Efeito estimado da chuva conforme a experiência do piloto (H3)",
                     fontweight="bold")
        fig.text(0.01, -0.02, "Modelos da Tabela M4: δ + θ·Experiência; faixa: IC 95% (erros agrupados por "
                 "corrida). Eixo horizontal em escala logarítmica.", fontsize=8, color=COR_TEXTO2)
        fig.tight_layout()
        salvar_figura(fig, "F12_H3_efeito_experiencia", "efeito da chuva conforme a experiência")

    # Figura 13: efeito da chuva no ritmo por piloto (análise complementar)
    t = tab_h3["Ritmo relativo ao grid (%)"].sort_values("Efeito")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.27 * len(t))))
    y = np.arange(len(t))
    ax.axvline(t["Efeito"].mean(), color=COR_NEUTRA, linewidth=1, linestyle="--",
               label="Média dos pilotos")
    ax.hlines(y, t["IC95% inf."], t["IC95% sup."], color=COR_MOLHADO, linewidth=1.5, alpha=0.5)
    ax.plot(t["Efeito"], y, "o", color=COR_MOLHADO, markersize=6)
    ax.set_yticks(y, t["Piloto"])
    ax.set_xlabel("Efeito da chuva no ritmo relativo ao grid (p.p.; negativo = ganha do grid)")
    ax.set_title("Figura 13 — Efeito da chuva por piloto (H3, análise complementar)")
    ax.legend(loc="lower right")
    salvar_figura(fig, "F13_H3_efeito_piloto", "efeito da chuva por piloto")


# =====================================================================
# EXECUÇÃO
# =====================================================================

def main():
    for d in (TAB_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)
    estilo_graficos()
    pc, voltas, t_media = carregar()
    m_h1 = hipotese_h1(pc)
    hipotese_h1_voltas(voltas)
    _, efeitos_h2 = hipotese_h2(pc)
    info_h3 = hipotese_h3(pc)
    tab_h3 = h3_complementar(pc)
    robustez(pc)
    figuras(m_h1, efeitos_h2, tab_h3, info_h3, t_media, pc)
    gravar_excel()
    with open(os.path.join(RES_DIR, "resumo_modelos.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(RESUMO))
    print("Modelos concluídos. Resultados em", RES_DIR)


if __name__ == "__main__":
    main()