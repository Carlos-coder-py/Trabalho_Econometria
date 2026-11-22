"""
PRÉ-REQUISITO
    Ter rodado src/01_coleta.py e src/02_tratamento.py (bases em dados/tratados/).
    Os parâmetros desta etapa (winsorização, mínimos de H3 etc.) ficam na
    seção 0 deste arquivo.

COMO RODAR (terminal do VS Code, na pasta raiz do projeto)
    python src/03_exploracao.py

O QUE ESTE CÓDIGO FAZ (cada bloco responde a uma parte da pesquisa)
    3.1  Carrega as bases tratadas e aplica a amostra principal
    3.2  Composição da amostra por temporada                 (Tabela 1)
    3.3  Estatísticas descritivas                            (Tabela 2)
    3.4  Distribuições, outliers e winsorização              (Tabela 3, Fig. 1)
    3.5  Viabilidade dos modelos: variação da temperatura,
         multicolinearidade e corridas molhadas por piloto   (Tabelas 4–6, Fig. 2)
    3.6  H1 — temperatura x desempenho e consistência        (Tabela 7, Fig. 3)
    3.7  H2 — chuva x erros (seco x molhado, com testes)     (Tabela 8, Fig. 4–5)
    3.8  H3 — heterogeneidade entre pilotos                  (Tabelas 9–10, Fig. 6–7)
    3.9  Matrizes de correlação (Pearson e Spearman)         (Tabelas 11–12, Fig. 8)
    3.10 Recorte por era do regulamento                      (Tabela 13)
    3.10b H3 — experiência do piloto x efeito da chuva        (Tabela 14, Fig. 9)
    3.11 Bases finais de análise para a etapa 4              (dados/final/)

SAÍDAS
    resultados/tabelas/tabelas_exploracao.xlsx -> todas as tabelas (uma por
        aba, com título e nota), prontas para montar as tabelas no Word
    resultados/tabelas/*.csv                   -> as mesmas tabelas em CSV
    resultados/figuras/*.png                   -> gráficos em 300 dpi
    resultados/resumo_exploracao.txt           -> números-chave da exploração
    dados/final/*.csv                          -> bases usadas na etapa 4

OBSERVAÇÃO SOBRE A INFERÊNCIA
    Os testes desta etapa são descritivos (comparações simples, sem
    controles). Eles orientam a etapa 4, onde os efeitos são estimados com
    controles, efeitos fixos e erros-padrão agrupados por corrida.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # gera os PNGs sem abrir janelas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# =====================================================================
# 0. CONFIGURAÇÃO
# =====================================================================

# Pasta raiz do projeto: a pasta acima de src/ (ou a própria pasta, se os
# códigos estiverem soltos na raiz). É onde ficam f1_cache/, dados/ e
# resultados/.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
DADOS_DIR = os.path.join(BASE_DIR, "dados")
TRATADOS_DIR = os.path.join(DADOS_DIR, "tratados")      # entrada (02_tratamento.py)
FINAL_DIR = os.path.join(DADOS_DIR, "final")            # bases para os modelos
RES_DIR = os.path.join(BASE_DIR, "resultados")
TAB_DIR = os.path.join(RES_DIR, "tabelas")
FIG_DIR = os.path.join(RES_DIR, "figuras")

# --- Decisões metodológicas desta etapa (citar na metodologia) ---
WINSOR_INF, WINSOR_SUP = 0.01, 0.99        # winsorização em 1% e 99%
MIN_CORRIDAS_MOLHADAS_H3 = 4               # piloto entra em H3 se tiver
MIN_CORRIDAS_SECAS_H3 = 10                 # >= 4 corridas molhadas e >= 10 secas
MIN_CORRIDAS_SENSIB_CALOR = 15             # mínimo de corridas secas p/ inclinação
N_FAIXAS_TEMPERATURA = 15                  # nº de faixas no gráfico de H1
ALFA = 0.05                                # nível de significância

# Paleta (cor por função): seco = laranja, molhado = azul, neutro = cinza
COR_SECO, COR_MOLHADO = "#eb6834", "#2a78d6"
COR_NEUTRA, COR_TEXTO, COR_TEXTO2 = "#8a8985", "#0b0b0b", "#52514e"

# Rótulos em português das variáveis (usados em tabelas e gráficos)
ROTULOS = {
    "AirTemp_mean": "Temperatura do ar (°C)",
    "TrackTemp_mean": "Temperatura da pista (°C)",
    "TrackTemp_max": "Temperatura máx. da pista (°C)",
    "Humidity_mean": "Umidade relativa (%)",
    "WindSpeed_mean": "Velocidade do vento (m/s)",
    "Rainfall_share_of_race": "Proporção do tempo com chuva",
    "ShareWetCompoundLaps_race": "Proporção de voltas com pneu de chuva",
    "PctTime_SafetyCarOrVSC": "Proporção do tempo sob SC/VSC",
    "TrackTemp_driver_mean": "Temperatura da pista nas voltas do piloto (°C)",
    "AvgLapTimePct_vsCircuit": "Ritmo vs. referência do circuito (%)",
    "AvgLapTimePct_vsRace": "Ritmo vs. mediana da corrida (%)",
    "LapTimePctStd": "Inconsistência: DP do ritmo (p.p.)",
    "ConsistencyResidStd": "Inconsistência: DP do resíduo no stint (p.p.)",
    "ErrorCount": "Erros (contagem)",
    "ErrorRate_per100laps": "Erros por 100 voltas",
    "NumDeletedLaps": "Voltas anuladas (limite de pista)",
    "NumPenalties_driving": "Penalidades de pilotagem",
    "NumSpinsMsg": "Rodadas (direção de prova)",
    "NumSlowLaps": "Voltas lentas suspeitas",
    "NumIncidents_driving": "Incidentes de pilotagem (envolvido)",
    "AnyDrivingPenalty": "Recebeu penalidade de pilotagem (0/1)",
    "DNF": "Abandono (0/1)",
    "DNF_Accident": "Abandono por acidente (0/1)",
    "GridPositionAdj": "Posição de largada",
    "GPsAnteriores": "Experiência: GPs disputados antes da corrida",
    "Idade": "Idade do piloto (anos)",
    "MolhadasAnteriores": "Corridas molhadas anteriores (desde 2021)",
    "NumLaps": "Voltas disputadas",
    "LapTimePct_vsCircuit": "Ritmo da volta vs. referência do circuito (%)",
    "LapTimePct_vsRace": "Ritmo da volta vs. mediana da corrida (%)",
    "TrackTemp": "Temperatura da pista na volta (°C)",
    "AirTemp": "Temperatura do ar na volta (°C)",
    "TyreLife": "Idade do pneu (voltas)",
}

TABELAS = []          # (nome_aba, título, DataFrame, nota) — gravadas no final
RESUMO = []           # linhas do resumo em texto


def log(msg):
    print(msg, flush=True)
    RESUMO.append(msg)


def rot(v):
    return ROTULOS.get(v, v)


def registrar_tabela(aba, titulo, df, nota=""):
    """Guarda a tabela para o Excel final e salva também em CSV."""
    TABELAS.append((aba, titulo, df, nota))
    df.to_csv(os.path.join(TAB_DIR, f"{aba}.csv"), index=False, encoding="utf-8-sig")


def gravar_excel():
    """Uma aba por tabela: título na linha 1, tabela a partir da linha 3 e
    nota explicativa logo abaixo — facilita montar as tabelas no Word."""
    caminho = os.path.join(TAB_DIR, "tabelas_exploracao.xlsx")
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


def estrelas(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_p(p):
    return "" if pd.isna(p) else ("<0,001" if p < 0.001 else f"{p:.3f}")


# Colunas que são SEMPRE numéricas (proporções, temperaturas, contagens).
# São convertidas à força: "True"/"False" viram 1/0 e qualquer outro texto
# vira vazio (NaN). Os valores estranhos encontrados são informados no terminal.
COLUNAS_NUMERICAS = [
    "ShareRainLaps", "ShareWetCompoundLaps", "ShareWetCompoundLaps_race",
    "Rainfall_share_of_race", "TrackTemp_driver_mean", "AirTemp_driver_mean",
    "TrackTemp_mean", "TrackTemp_max", "AirTemp_mean", "AirTemp_max",
    "Humidity_mean", "WindSpeed_mean", "PctTime_SafetyCarOrVSC", "PctTime_RedFlag",
    "TrackTemp", "AirTemp", "LapTimePct_vsCircuit", "LapTimePct_vsRace",
    "AvgLapTimePct_vsCircuit", "AvgLapTimePct_vsRace", "LapTimePctStd",
    "ConsistencyResidStd", "ErrorRate_per100laps", "ErrorCount", "NumLaps",
    "GridPositionAdj", "GPsAnteriores", "Idade", "MolhadasAnteriores",
]


def a_numero(serie, nome="", avisar=True):
    """Converte uma coluna para número de forma segura."""
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        return serie.astype(float)
    texto = serie.astype("string").str.strip()
    mapa = {"TRUE": "1", "FALSE": "0"}
    texto = texto.str.upper().replace(mapa)
    num = pd.to_numeric(texto, errors="coerce")
    estranhos = texto[num.isna() & texto.notna() & ~texto.isin(["", "NAN", "<NA>", "NONE"])]
    if avisar and len(estranhos):
        exemplos = ", ".join(map(str, estranhos.unique()[:5]))
        log(f"  aviso: {len(estranhos)} valores não numéricos em '{nome}' viraram vazio "
            f"(exemplos: {exemplos})")
    return num.astype(float)


def garantir_numericas(df, minimo=0.9):
    """Converte para número as colunas de texto em que ao menos 90% dos
    valores preenchidos são números (ex.: 0.25, 1.0, "<NA>" -> NaN).
    Colunas de texto de verdade (piloto, equipe, True/False) não mudam."""
    for c in COLUNAS_NUMERICAS:
        if c in df.columns:
            df[c] = a_numero(df[c], c)
    texto = [c for c in df.columns if c not in COLUNAS_NUMERICAS
             and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))]
    for c in texto:
        preenchidos = df[c].notna().sum()
        if preenchidos == 0:
            continue
        convertida = pd.to_numeric(df[c], errors="coerce")
        if convertida.notna().sum() >= minimo * preenchidos:
            df[c] = convertida


def como_bool(s):
    return s.astype(str).str.strip().str.upper().isin(["TRUE", "1", "1.0"])


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
    caminho = os.path.join(FIG_DIR, f"{nome}.png")
    fig.savefig(caminho)
    plt.close(fig)
    log(f"  figura: {nome}.png — {titulo}")


def remover_efeitos_fixos(df, y, efeitos, iteracoes=30):
    """Retira de y as médias dos grupos indicados (ex.: circuito e
    equipe-temporada) por demeaning alternado. Serve para visualizar a
    relação de interesse 'por dentro' do circuito e do carro, como fará o
    modelo de efeitos fixos da etapa 4."""
    r = df[y] - df[y].mean()
    for _ in range(iteracoes):
        for fe in efeitos:
            r = r - r.groupby(df[fe]).transform("mean")
    return r + df[y].mean()


def descritivas(df, variaveis):
    linhas = []
    for v in variaveis:
        s = pd.to_numeric(df[v], errors="coerce").dropna()
        if s.empty:
            continue
        linhas.append({"Variável": rot(v), "N": len(s), "Média": s.mean(),
                       "Desvio-padrão": s.std(), "Mínimo": s.min(),
                       "P25": s.quantile(.25), "Mediana": s.median(),
                       "P75": s.quantile(.75), "Máximo": s.max(),
                       "Assimetria": stats.skew(s)})
    return pd.DataFrame(linhas).round(3)


# =====================================================================
# 3.1 CARREGAR AS BASES E APLICAR A AMOSTRA PRINCIPAL
# =====================================================================

def carregar():
    log("3.1 Carregando as bases tratadas...")
    voltas = pd.read_csv(os.path.join(TRATADOS_DIR, "voltas_tratadas.csv"), low_memory=False)
    pc = pd.read_csv(os.path.join(TRATADOS_DIR, "piloto_corrida.csv"), low_memory=False)
    corr = pd.read_csv(os.path.join(TRATADOS_DIR, "corridas.csv"), low_memory=False)

    # Colunas numéricas que o pandas leu como texto (por causa de algum valor
    # não numérico, como "<NA>") são convertidas para número; o que não for
    # número vira vazio (NaN).
    for df in (voltas, pc, corr):
        garantir_numericas(df)

    for df, cols in [
        (voltas, ["AmostraPrincipal", "IsPaceLap", "IsCleanLap", "IsWetCompound",
                  "Rainfall", "Deleted", "IsSlowLap"]),
        (pc, ["AmostraPrincipal", "DNF", "DNF_Accident", "Finished", "WetRace",
              "AnyDrivingPenalty", "AnySeriousError"]),
        (corr, ["AmostraPrincipal", "WetRace", "Rainfall_any"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = como_bool(df[c])

    excl = corr.loc[~corr["AmostraPrincipal"], ["RaceId", "EventName", "MotivoExclusao"]]
    for _, r in excl.iterrows():
        log(f"  excluída da amostra principal: {r['RaceId']} {r['EventName']} — {r['MotivoExclusao']}")
    voltas = voltas[voltas["AmostraPrincipal"]].copy()
    pc = pc[pc["AmostraPrincipal"]].copy()
    corr = corr[corr["AmostraPrincipal"]].copy()

    # Indicadores auxiliares
    pc["Condicao"] = np.where(pc["WetRace"], "Molhada", "Seca")
    corr["Condicao"] = np.where(corr["WetRace"], "Molhada", "Seca")
    pc["DNF_num"] = pc["DNF"].astype(int)
    pc["DNF_Accident_num"] = pc["DNF_Accident"].astype(int)
    pc["AnyDrivingPenalty_num"] = pc["AnyDrivingPenalty"].astype(int)
    log(f"  amostra principal: {corr['RaceId'].nunique()} corridas, {len(pc)} "
        f"observações piloto-corrida, {int(voltas['IsPaceLap'].sum())} voltas de ritmo.")
    return voltas, pc, corr


# =====================================================================
# 3.2 COMPOSIÇÃO DA AMOSTRA (Tabela 1)
# =====================================================================

def composicao_amostra(voltas, pc, corr):
    log("3.2 Composição da amostra...")
    t = corr.groupby("Season").agg(Corridas=("RaceId", "nunique"),
                                   Corridas_molhadas=("WetRace", "sum"))
    t["Pilotos"] = pc.groupby("Season")["Driver"].nunique()
    t["Obs_piloto_corrida"] = pc.groupby("Season").size()
    t["Voltas_de_ritmo"] = voltas[voltas["IsPaceLap"]].groupby("Season").size()
    t["Abandonos"] = pc.groupby("Season")["DNF"].sum()
    t["Abandonos_por_acidente"] = pc.groupby("Season")["DNF_Accident"].sum()
    fonte = pc[pc["DNF"]].groupby("Season")["DNF_Accident_Source"].agg(
        lambda s: "oficial" if s.str.startswith("oficial").mean() > .5 else "aproximação")
    t["Fonte_causa_abandono"] = fonte
    t = t.reset_index().rename(columns={"Season": "Temporada"})
    total = t.drop(columns=["Temporada", "Fonte_causa_abandono"]).sum()
    total["Pilotos"] = pc["Driver"].nunique()
    t.loc[len(t)] = ["Total", *total.values, ""]
    registrar_tabela("T01_amostra", "Tabela 1 — Composição da amostra por temporada", t,
                     "Amostra principal. Corrida molhada: >= 10% das voltas com pneu de chuva "
                     "ou >= 10% do tempo com chuva. Causa do abandono oficial em 2021–2022; "
                     "aproximação validada nas demais temporadas.")


# =====================================================================
# 3.3 ESTATÍSTICAS DESCRITIVAS (Tabela 2)
# =====================================================================

VARS_CORRIDA = ["TrackTemp_mean", "AirTemp_mean", "Humidity_mean", "WindSpeed_mean",
                "Rainfall_share_of_race", "ShareWetCompoundLaps", "PctTime_SafetyCarOrVSC"]
VARS_PC = ["AvgLapTimePct_vsCircuit", "AvgLapTimePct_vsRace", "LapTimePctStd",
           "ConsistencyResidStd", "ErrorRate_per100laps", "ErrorCount", "NumDeletedLaps",
           "NumPenalties_driving", "NumSpinsMsg", "NumSlowLaps", "DNF_num",
           "DNF_Accident_num", "GridPositionAdj", "NumLaps", "GPsAnteriores", "Idade"]
VARS_VOLTA = ["LapTimePct_vsCircuit", "LapTimePct_vsRace", "TrackTemp", "AirTemp", "TyreLife"]


def estatisticas_descritivas(voltas, pc, corr):
    log("3.3 Estatísticas descritivas...")
    ROTULOS.update({"ShareWetCompoundLaps": "Proporção de voltas com pneu de chuva",
                    "DNF_num": "Abandono (0/1)", "DNF_Accident_num": "Abandono por acidente (0/1)"})
    partes = []
    for nivel, df, vs in [("Corrida", corr, VARS_CORRIDA), ("Piloto-corrida", pc, VARS_PC),
                          ("Volta (voltas de ritmo)", voltas[voltas["IsPaceLap"]], VARS_VOLTA)]:
        d = descritivas(df, [v for v in vs if v in df.columns])
        d.insert(0, "Nível", nivel)
        partes.append(d)
    registrar_tabela("T02_descritivas", "Tabela 2 — Estatísticas descritivas",
                     pd.concat(partes, ignore_index=True),
                     "P25 e P75: 1º e 3º quartis. Voltas de ritmo: voltas sem box, sem 1ª volta, "
                     "sem SC/VSC/bandeira vermelha e não lentas.")


# =====================================================================
# 3.4 DISTRIBUIÇÕES, OUTLIERS E WINSORIZAÇÃO (Tabela 3, Figura 1)
# =====================================================================

VARS_WINSOR_PC = ["AvgLapTimePct_vsCircuit", "AvgLapTimePct_vsRace", "LapTimePctStd",
                  "ConsistencyResidStd", "ErrorRate_per100laps"]


def winsorizar(s, inf=WINSOR_INF, sup=WINSOR_SUP):
    lo, hi = s.quantile(inf), s.quantile(sup)
    return s.clip(lo, hi), lo, hi


def outliers_e_winsorizacao(voltas, pc):
    log("3.4 Distribuições, outliers e winsorização...")
    linhas = []
    for v in VARS_WINSOR_PC:
        s = pc[v]
        w, lo, hi = winsorizar(s)
        pc[v + "_w"] = w
        linhas.append({"Variável": rot(v), "Base": "Piloto-corrida", "N": s.notna().sum(),
                       "P1": lo, "P99": hi, "% abaixo de P1": 100 * (s < lo).mean(),
                       "% acima de P99": 100 * (s > hi).mean(),
                       "Assimetria original": stats.skew(s.dropna()),
                       "Assimetria winsorizada": stats.skew(w.dropna())})
    ritmo = voltas["IsPaceLap"]
    for v in ["LapTimePct_vsCircuit", "LapTimePct_vsRace"]:
        s = voltas.loc[ritmo, v]
        w, lo, hi = winsorizar(s)
        voltas.loc[ritmo, v + "_w"] = w
        linhas.append({"Variável": rot(v), "Base": "Volta (ritmo)", "N": s.notna().sum(),
                       "P1": lo, "P99": hi, "% abaixo de P1": 100 * (s < lo).mean(),
                       "% acima de P99": 100 * (s > hi).mean(),
                       "Assimetria original": stats.skew(s.dropna()),
                       "Assimetria winsorizada": stats.skew(w.dropna())})
    registrar_tabela("T03_winsorizacao", "Tabela 3 — Valores extremos e winsorização (1% e 99%)",
                     pd.DataFrame(linhas).round(3),
                     "Valores abaixo do percentil 1 e acima do percentil 99 substituídos pelos "
                     "próprios percentis (variáveis com sufixo _w nas bases finais). Contagens de "
                     "erro não são winsorizadas.")

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    pares = [(voltas.loc[ritmo, "LapTimePct_vsCircuit"], rot("LapTimePct_vsCircuit")),
             (pc["LapTimePctStd"], rot("LapTimePctStd")),
             (pc["ErrorRate_per100laps"], rot("ErrorRate_per100laps")),
             (voltas.loc[ritmo, "TrackTemp"], rot("TrackTemp"))]
    for a, (s, nome) in zip(ax.flat, pares):
        s = s.dropna()
        lo, hi = s.quantile(WINSOR_INF), s.quantile(WINSOR_SUP)
        a.hist(s.clip(s.quantile(.001), s.quantile(.999)), bins=50, color=COR_NEUTRA,
               edgecolor="white", linewidth=0.4)
        for q in (lo, hi):
            a.axvline(q, color=COR_SECO, linestyle="--", linewidth=1.2)
        a.set_title(nome, fontsize=10)
        a.set_ylabel("Frequência")
    ax.flat[0].plot([], [], color=COR_SECO, linestyle="--", label="Percentis 1 e 99")
    ax.flat[0].legend(loc="upper right")
    fig.suptitle("Figura 1 — Distribuição das principais variáveis", fontweight="bold")
    fig.tight_layout()
    salvar_figura(fig, "F01_distribuicoes", "distribuições e limites de winsorização")


# =====================================================================
# 3.5 VIABILIDADE DOS MODELOS (Tabelas 4–6, Figura 2)
# =====================================================================

def viabilidade(voltas, pc, corr):
    log("3.5 Testes de viabilidade dos modelos...")
    linhas = []

    # (a) Quanto da variação da temperatura está DENTRO dos circuitos?
    #     (com efeito fixo de circuito, só essa parte identifica H1)
    for v in ["TrackTemp_mean", "AirTemp_mean"]:
        d = corr.dropna(subset=[v])
        r2 = smf.ols(f"{v} ~ C(Location)", data=d).fit().rsquared
        linhas.append({"Indicador": f"Parcela da variância de '{rot(v)}' dentro dos circuitos",
                       "Valor": 1 - r2, "Base": "Corrida",
                       "Leitura": "1 − R² da regressão da temperatura em dummies de circuito"})
    d = voltas[voltas["IsPaceLap"]].dropna(subset=["TrackTemp"])
    dentro = 1 - d.groupby("RaceId")["TrackTemp"].transform("mean").var() / d["TrackTemp"].var()
    linhas.append({"Indicador": "Parcela da variância da temperatura da pista dentro da corrida",
                   "Valor": dentro, "Base": "Volta",
                   "Leitura": "variação ao longo da própria corrida (entre voltas)"})

    # (b) Correlação ar x pista e multicolinearidade (VIF) entre as variáveis climáticas
    r_corr = corr[["AirTemp_mean", "TrackTemp_mean"]].corr().iloc[0, 1]
    linhas.append({"Indicador": "Correlação entre temperatura do ar e da pista", "Valor": r_corr,
                   "Base": "Corrida", "Leitura": "Pearson"})
    clima = ["TrackTemp_mean", "AirTemp_mean", "Humidity_mean", "WindSpeed_mean",
             "Rainfall_share_of_race"]
    X = corr[clima].dropna()
    X = X.assign(const=1.0)
    for i, v in enumerate(clima):
        linhas.append({"Indicador": f"VIF — {rot(v)}", "Valor": variance_inflation_factor(X.values, i),
                       "Base": "Corrida", "Leitura": "VIF > 10 indica multicolinearidade severa"})
    registrar_tabela("T04_viabilidade", "Tabela 4 — Variação da temperatura e multicolinearidade",
                     pd.DataFrame(linhas).round(3),
                     "VIF calculado com todas as variáveis climáticas da corrida no mesmo modelo.")
    log(f"  correlação ar x pista = {r_corr:.2f}; parcela da variância da temperatura da pista "
        f"dentro dos circuitos = {linhas[0]['Valor']:.2f}")

    # (c) Temperatura por circuito (Tabela 5 e Figura 2)
    t5 = (corr.groupby("Location")["TrackTemp_mean"]
          .agg(Corridas="count", Media="mean", DP="std", Minimo="min", Maximo="max")
          .assign(Amplitude=lambda x: x["Maximo"] - x["Minimo"])
          .sort_values("Media").round(2).reset_index().rename(columns={"Location": "Circuito"}))
    registrar_tabela("T05_temp_por_circuito", "Tabela 5 — Temperatura média da pista por circuito",
                     t5, "Temperatura média da pista durante a corrida (°C), 2021–2026.")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(t5))))
    y = np.arange(len(t5))
    ax.hlines(y, t5["Minimo"], t5["Maximo"], color="#c3c2b7", linewidth=2)
    ax.plot(t5["Media"], y, "o", color=COR_SECO, markersize=6, label="Média")
    ax.plot(t5["Minimo"], y, "|", color=COR_TEXTO2, markersize=8)
    ax.plot(t5["Maximo"], y, "|", color=COR_TEXTO2, markersize=8, label="Mínimo e máximo")
    ax.set_yticks(y, t5["Circuito"])
    ax.set_xlabel(rot("TrackTemp_mean"))
    ax.set_title("Figura 2 — Variação da temperatura da pista por circuito")
    ax.legend(loc="lower right")
    salvar_figura(fig, "F02_temperatura_por_circuito", "amplitude térmica por circuito")

    # (d) Corridas secas e molhadas por piloto (Tabela 6) — viabilidade de H3
    t6 = pc.pivot_table(index="Driver", columns="Condicao", values="RaceId",
                        aggfunc="count", fill_value=0)
    for c in ("Seca", "Molhada"):
        if c not in t6.columns:
            t6[c] = 0
    t6 = t6[["Seca", "Molhada"]].rename(columns={"Seca": "Corridas_secas",
                                                 "Molhada": "Corridas_molhadas"})
    t6["Elegivel_H3"] = (t6["Corridas_molhadas"] >= MIN_CORRIDAS_MOLHADAS_H3) & \
                        (t6["Corridas_secas"] >= MIN_CORRIDAS_SECAS_H3)
    t6 = t6.sort_values("Corridas_molhadas", ascending=False).reset_index().rename(
        columns={"Driver": "Piloto"})
    registrar_tabela("T06_pilotos_seco_molhado", "Tabela 6 — Corridas secas e molhadas por piloto",
                     t6, f"Elegível para H3: >= {MIN_CORRIDAS_MOLHADAS_H3} corridas molhadas e "
                         f">= {MIN_CORRIDAS_SECAS_H3} secas.")
    log(f"  pilotos elegíveis para H3: {int(t6['Elegivel_H3'].sum())} de {len(t6)}")
    return set(t6.loc[t6["Elegivel_H3"], "Piloto"])


# =====================================================================
# 3.6 H1 — TEMPERATURA x DESEMPENHO E CONSISTÊNCIA (Tabela 7, Figura 3)
# =====================================================================

def grafico_faixas(ax, x, y, cor, n_faixas=N_FAIXAS_TEMPERATURA, rotulo=None):
    """Média de y em faixas de x com mesmo nº de observações (binscatter),
    intervalo de confiança de 95% e ajuste quadrático sobre os dados."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    d["faixa"] = pd.qcut(d["x"], n_faixas, duplicates="drop")
    g = d.groupby("faixa", observed=True).agg(x=("x", "mean"), y=("y", "mean"),
                                              dp=("y", "std"), n=("y", "count"))
    ic = 1.96 * g["dp"] / np.sqrt(g["n"])
    ax.errorbar(g["x"], g["y"], yerr=ic, fmt="o", color=cor, ecolor=cor, elinewidth=1,
                capsize=0, markersize=6, label=rotulo)
    coef = np.polyfit(d["x"], d["y"], 2)
    xs = np.linspace(d["x"].quantile(.01), d["x"].quantile(.99), 100)
    ax.plot(xs, np.polyval(coef, xs), color=cor, alpha=0.6)
    return coef


def hipotese_h1(pc):
    log("3.6 H1 — temperatura x desempenho...")
    seco = pc[~pc["WetRace"]].dropna(subset=["TrackTemp_driver_mean",
                                              "AvgLapTimePct_vsCircuit_w"]).copy()
    # Ritmo e inconsistência "por dentro" do circuito e do carro
    seco["ritmo_res"] = remover_efeitos_fixos(seco, "AvgLapTimePct_vsCircuit_w",
                                              ["Location", "TeamSeason"])
    seco["incons_res"] = remover_efeitos_fixos(seco.dropna(subset=["LapTimePctStd_w"]),
                                               "LapTimePctStd_w", ["Location", "TeamSeason"])

    # Tabela 7: ritmo e inconsistência por quintil de temperatura da pista
    seco["Quintil"] = pd.qcut(seco["TrackTemp_driver_mean"], 5,
                              labels=["Q1 (mais fria)", "Q2", "Q3", "Q4", "Q5 (mais quente)"])
    t7 = seco.groupby("Quintil", observed=True).agg(
        N=("Driver", "count"),
        Temp_min=("TrackTemp_driver_mean", "min"), Temp_max=("TrackTemp_driver_mean", "max"),
        Ritmo_medio=("AvgLapTimePct_vsCircuit_w", "mean"),
        Ritmo_residual=("ritmo_res", "mean"),
        Inconsistencia_media=("LapTimePctStd_w", "mean"),
        Inconsistencia_residual=("incons_res", "mean")).round(3).reset_index()
    grupos_r = [g["ritmo_res"].dropna() for _, g in seco.groupby("Quintil", observed=True)]
    grupos_i = [g["incons_res"].dropna() for _, g in seco.groupby("Quintil", observed=True)]
    kw_r, kw_i = stats.kruskal(*grupos_r), stats.kruskal(*grupos_i)
    rp = stats.pearsonr(seco["TrackTemp_driver_mean"], seco["ritmo_res"])
    rs = stats.spearmanr(seco["TrackTemp_driver_mean"], seco["ritmo_res"])
    nota = (f"Corridas secas. Residual = após retirar médias de circuito e de equipe-temporada. "
            f"Kruskal-Wallis entre quintis: ritmo H = {kw_r.statistic:.2f} (p = {fmt_p(kw_r.pvalue)}); "
            f"inconsistência H = {kw_i.statistic:.2f} (p = {fmt_p(kw_i.pvalue)}). "
            f"Correlação temperatura x ritmo residual: Pearson r = {rp[0]:.3f} (p = {fmt_p(rp[1])}); "
            f"Spearman ρ = {rs[0]:.3f} (p = {fmt_p(rs[1])}).")
    registrar_tabela("T07_H1_quintis_temperatura",
                     "Tabela 7 — Ritmo e inconsistência por quintil de temperatura da pista (H1)",
                     t7, nota)
    log(f"  H1: Kruskal-Wallis ritmo p = {fmt_p(kw_r.pvalue)}; inconsistência p = {fmt_p(kw_i.pvalue)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    c1 = grafico_faixas(ax[0], seco["TrackTemp_driver_mean"], seco["ritmo_res"], COR_SECO)
    ax[0].set_xlabel(rot("TrackTemp"))
    ax[0].set_ylabel("Ritmo residual (% vs. referência)")
    ax[0].set_title("(a) Ritmo médio")
    grafico_faixas(ax[1], seco["TrackTemp_driver_mean"], seco["incons_res"], COR_SECO)
    ax[1].set_xlabel(rot("TrackTemp"))
    ax[1].set_ylabel("Inconsistência residual (p.p.)")
    ax[1].set_title("(b) Inconsistência")
    fig.suptitle("Figura 3 — Temperatura da pista, ritmo e inconsistência (corridas secas)",
                 fontweight="bold")
    fig.text(0.01, -0.03, "Pontos: média em faixas com o mesmo nº de observações (IC 95%). "
             "Linha: ajuste quadrático. Valores após retirar médias de circuito e equipe-temporada.",
             fontsize=8, color=COR_TEXTO2)
    fig.tight_layout()
    salvar_figura(fig, "F03_H1_temperatura_ritmo", "binscatter temperatura x ritmo/inconsistência")
    if c1[0] != 0:
        tipo = "mínimo (formato em U)" if c1[0] > 0 else "máximo (formato em U invertido)"
        log(f"  ajuste quadrático do ritmo: {tipo} em {-c1[1] / (2 * c1[0]):.1f} °C "
            f"(coeficiente do termo quadrático = {c1[0]:.5f}); faixa observada: "
            f"{seco['TrackTemp_driver_mean'].min():.1f} a {seco['TrackTemp_driver_mean'].max():.1f} °C")


# =====================================================================
# 3.7 H2 — CHUVA x ERROS (Tabela 8, Figuras 4–5)
# =====================================================================

VARS_H2_CONT = ["ErrorRate_per100laps", "ErrorCount", "NumDeletedLaps", "NumPenalties_driving",
                "NumSpinsMsg", "NumSlowLaps", "NumIncidents_driving", "LapTimePctStd",
                "ConsistencyResidStd"]
VARS_H2_BIN = ["AnyDrivingPenalty", "DNF", "DNF_Accident"]


def hipotese_h2(pc):
    log("3.7 H2 — chuva x erros...")
    seco, molh = pc[~pc["WetRace"]], pc[pc["WetRace"]]
    linhas = []
    for v in VARS_H2_CONT:
        a, b = seco[v].dropna(), molh[v].dropna()
        t = stats.ttest_ind(b, a, equal_var=False)
        u = stats.mannwhitneyu(b, a, alternative="two-sided")
        linhas.append({"Variável": rot(v), "Média seca": a.mean(), "Média molhada": b.mean(),
                       "Diferença (molhada − seca)": b.mean() - a.mean(),
                       "Teste": "t de Welch / Mann-Whitney",
                       "Estatística": t.statistic, "p-valor (t)": fmt_p(t.pvalue),
                       "p-valor (Mann-Whitney)": fmt_p(u.pvalue), "Sig.": estrelas(t.pvalue),
                       "N seca": len(a), "N molhada": len(b)})
    for v in VARS_H2_BIN:
        tab = pd.crosstab(pc["WetRace"], pc[v])
        chi = stats.chi2_contingency(tab) if tab.shape == (2, 2) else (np.nan, np.nan)
        linhas.append({"Variável": rot(v), "Média seca": seco[v].mean(), "Média molhada": molh[v].mean(),
                       "Diferença (molhada − seca)": molh[v].mean() - seco[v].mean(),
                       "Teste": "Qui-quadrado", "Estatística": chi[0], "p-valor (t)": fmt_p(chi[1]),
                       "p-valor (Mann-Whitney)": "", "Sig.": estrelas(chi[1]),
                       "N seca": len(seco), "N molhada": len(molh)})
    t8 = pd.DataFrame(linhas)
    for c in ["Média seca", "Média molhada", "Diferença (molhada − seca)", "Estatística"]:
        t8[c] = t8[c].astype(float).round(3)
    registrar_tabela("T08_H2_seco_molhado", "Tabela 8 — Erros e consistência em corridas secas e molhadas (H2)",
                     t8, "Base piloto-corrida. Variáveis 0/1: média = proporção. *** p<0,01; "
                         "** p<0,05; * p<0,10. Comparação simples, sem controles (ver etapa 4). "
                         "Abandono por acidente: causa oficial em 2021–2022 e aproximação nas demais.")
    lin = t8.set_index("Variável")
    log(f"  H2: erros por 100 voltas — seca {lin.loc[rot('ErrorRate_per100laps'), 'Média seca']:.2f}, "
        f"molhada {lin.loc[rot('ErrorRate_per100laps'), 'Média molhada']:.2f} "
        f"(p = {lin.loc[rot('ErrorRate_per100laps'), 'p-valor (t)']})")

    # Figura 4: pequenos múltiplos (cada medida com sua própria escala)
    medidas = ["ErrorRate_per100laps", "NumSlowLaps", "AnyDrivingPenalty", "DNF_Accident"]
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.6))
    for a, v in zip(ax, medidas):
        for i, (nome, df, cor) in enumerate([("Seca", seco, COR_SECO), ("Molhada", molh, COR_MOLHADO)]):
            s = df[v].astype(float).dropna()
            m, ic = s.mean(), 1.96 * s.std() / np.sqrt(len(s))
            a.bar(i, m, width=0.6, color=cor, edgecolor="white")
            a.errorbar(i, m, yerr=ic, color=COR_TEXTO2, capsize=3, linewidth=1)
        a.set_xticks([0, 1], ["Seca", "Molhada"])
        a.set_title(rot(v).replace(" (0/1)", ""), fontsize=9)
        a.grid(axis="x", visible=False)
    fig.suptitle("Figura 4 — Erros em corridas secas e molhadas (média e IC 95%)", fontweight="bold")
    fig.tight_layout()
    salvar_figura(fig, "F04_H2_seco_molhado", "comparação de erros seco x molhado")

    # Figura 5: taxa de erro conforme a proporção de voltas com chuva vividas pelo piloto
    d = pc.copy()
    d["ShareRainLaps"] = a_numero(d["ShareRainLaps"], "ShareRainLaps", avisar=False)
    d = d.dropna(subset=["ShareRainLaps", "ErrorRate_per100laps_w"])
    d["Faixa_chuva"] = pd.cut(d["ShareRainLaps"], [-0.001, 0, 0.25, 0.5, 1.0],
                              labels=["0%", "0–25%", "25–50%", "50–100%"])
    g = d.groupby("Faixa_chuva", observed=True)["ErrorRate_per100laps_w"].agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], width=0.6, color=[COR_SECO] + [COR_MOLHADO] * (len(g) - 1), edgecolor="white")
    ax.errorbar(x, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]), fmt="none",
                ecolor=COR_TEXTO2, capsize=3, linewidth=1)
    for xi, n in zip(x, g["count"]):
        ax.text(xi, 0, f"n={n}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x, g.index.astype(str))
    ax.set_xlabel("Proporção das voltas do piloto com chuva")
    ax.set_ylabel(rot("ErrorRate_per100laps"))
    ax.set_title("Figura 5 — Taxa de erro conforme a exposição à chuva")
    ax.grid(axis="x", visible=False)
    salvar_figura(fig, "F05_H2_exposicao_chuva", "taxa de erro por faixa de chuva")


# =====================================================================
# 3.8 H3 — HETEROGENEIDADE ENTRE PILOTOS (Tabelas 9–10, Figuras 6–7)
# =====================================================================

def dif_com_ic(a, b):
    """Diferença de médias (b − a) com IC 95% de Welch."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    dif = b.mean() - a.mean()
    ep = np.sqrt(a.var() / len(a) + b.var() / len(b))
    gl = ep ** 4 / ((a.var() / len(a)) ** 2 / (len(a) - 1) + (b.var() / len(b)) ** 2 / (len(b) - 1))
    tc = stats.t.ppf(0.975, gl)
    return dif, dif - tc * ep, dif + tc * ep


def teste_interacao(df, y):
    """Teste F conjunto das interações piloto × corrida molhada: os efeitos
    da chuva diferem entre os pilotos?"""
    d = df.dropna(subset=[y])
    m0 = smf.ols(f"{y} ~ C(Driver) + WetRace", data=d).fit()
    m1 = smf.ols(f"{y} ~ C(Driver) * WetRace", data=d).fit()
    a = anova_lm(m0, m1)
    return a["F"].iloc[1], a["Pr(>F)"].iloc[1], a["df_diff"].iloc[1]


def hipotese_h3(pc, elegiveis):
    log("3.8 H3 — heterogeneidade entre pilotos...")
    d = pc[pc["Driver"].isin(elegiveis)].copy()
    if d.empty:
        log("  nenhum piloto elegível para H3 — tabelas não geradas.")
        return
    d["WetRace"] = d["WetRace"].astype(int)
    linhas = []
    for p, g in d.groupby("Driver"):
        s, m = g[g["WetRace"] == 0], g[g["WetRace"] == 1]
        dr, dr_lo, dr_hi = dif_com_ic(s["AvgLapTimePct_vsRace_w"], m["AvgLapTimePct_vsRace_w"])
        de, de_lo, de_hi = dif_com_ic(s["ErrorRate_per100laps_w"], m["ErrorRate_per100laps_w"])
        linhas.append({"Piloto": p, "Corridas secas": len(s), "Corridas molhadas": len(m),
                       "Ritmo seco (%)": s["AvgLapTimePct_vsRace_w"].mean(),
                       "Ritmo molhado (%)": m["AvgLapTimePct_vsRace_w"].mean(),
                       "Dif. ritmo (molh. − seco)": dr, "IC95% inf. (ritmo)": dr_lo,
                       "IC95% sup. (ritmo)": dr_hi,
                       "Erros/100v seco": s["ErrorRate_per100laps_w"].mean(),
                       "Erros/100v molhado": m["ErrorRate_per100laps_w"].mean(),
                       "Dif. erros (molh. − seco)": de, "IC95% inf. (erros)": de_lo,
                       "IC95% sup. (erros)": de_hi})
    t9 = pd.DataFrame(linhas).sort_values("Dif. ritmo (molh. − seco)").round(3)
    f_r, p_r, gl_r = teste_interacao(d, "AvgLapTimePct_vsRace_w")
    f_e, p_e, gl_e = teste_interacao(d, "ErrorRate_per100laps_w")
    registrar_tabela("T09_H3_pilotos_chuva",
                     "Tabela 9 — Diferença de ritmo e de erros na chuva, por piloto (H3)", t9,
                     f"Pilotos com >= {MIN_CORRIDAS_MOLHADAS_H3} corridas molhadas e >= "
                     f"{MIN_CORRIDAS_SECAS_H3} secas. Ritmo relativo à mediana da corrida (valores "
                     f"negativos = mais rápido que o grid). Teste F conjunto das interações "
                     f"piloto × corrida molhada: ritmo F({gl_r:.0f}) = {f_r:.2f}, p = {fmt_p(p_r)}; "
                     f"erros F({gl_e:.0f}) = {f_e:.2f}, p = {fmt_p(p_e)}.")
    log(f"  H3: teste F das interações piloto × chuva — ritmo p = {fmt_p(p_r)}; erros p = {fmt_p(p_e)}")

    # Figura 6: diferença de ritmo na chuva por piloto (dot plot com IC)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(t9))))
    y = np.arange(len(t9))
    ax.axvline(0, color=COR_NEUTRA, linewidth=1)
    ax.hlines(y, t9["IC95% inf. (ritmo)"], t9["IC95% sup. (ritmo)"], color=COR_MOLHADO,
              linewidth=1.5, alpha=0.5)
    ax.plot(t9["Dif. ritmo (molh. − seco)"], y, "o", color=COR_MOLHADO, markersize=6)
    ax.set_yticks(y, t9["Piloto"])
    ax.set_xlabel("Ritmo na chuva − ritmo no seco (p.p.; negativo = ganha do grid na chuva)")
    ax.set_title("Figura 6 — Desempenho relativo na chuva por piloto (IC 95%)")
    salvar_figura(fig, "F06_H3_pilotos_chuva", "diferença de ritmo molhado − seco por piloto")

    # Tabela 10 e Figura 7: sensibilidade ao calor por piloto (corridas secas)
    seco = pc[~pc["WetRace"]].dropna(subset=["TrackTemp_driver_mean", "AvgLapTimePct_vsRace_w"])
    linhas = []
    for p, g in seco.groupby("Driver"):
        if len(g) < MIN_CORRIDAS_SENSIB_CALOR:
            continue
        r = stats.linregress(g["TrackTemp_driver_mean"], g["AvgLapTimePct_vsRace_w"])
        tc = stats.t.ppf(0.975, len(g) - 2)
        linhas.append({"Piloto": p, "Corridas secas": len(g), "Inclinação (p.p./°C)": r.slope,
                       "IC95% inf.": r.slope - tc * r.stderr, "IC95% sup.": r.slope + tc * r.stderr,
                       "p-valor": fmt_p(r.pvalue)})
    if linhas:
        t10 = pd.DataFrame(linhas).sort_values("Inclinação (p.p./°C)").round(4)
        d2 = seco[seco["Driver"].isin(t10["Piloto"])]
        m0 = smf.ols("AvgLapTimePct_vsRace_w ~ C(Driver) + TrackTemp_driver_mean", data=d2).fit()
        m1 = smf.ols("AvgLapTimePct_vsRace_w ~ C(Driver) * TrackTemp_driver_mean", data=d2).fit()
        a = anova_lm(m0, m1)
        registrar_tabela("T10_H3_sensibilidade_calor",
                         "Tabela 10 — Sensibilidade do ritmo relativo à temperatura, por piloto (H3)",
                         t10, f"Corridas secas; pilotos com >= {MIN_CORRIDAS_SENSIB_CALOR} corridas. "
                              f"Inclinação da regressão simples do ritmo relativo (%) na temperatura "
                              f"da pista. Teste F conjunto das interações piloto × temperatura: "
                              f"F({a['df_diff'].iloc[1]:.0f}) = {a['F'].iloc[1]:.2f}, "
                              f"p = {fmt_p(a['Pr(>F)'].iloc[1])}.")
        log(f"  H3: teste F das interações piloto × temperatura — p = {fmt_p(a['Pr(>F)'].iloc[1])}")
        fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(t10))))
        y = np.arange(len(t10))
        ax.axvline(0, color=COR_NEUTRA, linewidth=1)
        ax.hlines(y, t10["IC95% inf."], t10["IC95% sup."], color=COR_SECO, linewidth=1.5, alpha=0.5)
        ax.plot(t10["Inclinação (p.p./°C)"], y, "o", color=COR_SECO, markersize=6)
        ax.set_yticks(y, t10["Piloto"])
        ax.set_xlabel("Variação do ritmo relativo por °C de pista (p.p.; positivo = perde rendimento no calor)")
        ax.set_title("Figura 7 — Sensibilidade ao calor por piloto (IC 95%)")
        salvar_figura(fig, "F07_H3_sensibilidade_calor", "inclinação ritmo x temperatura por piloto")


# =====================================================================
# 3.9 MATRIZES DE CORRELAÇÃO (Tabelas 11–12, Figura 8)
# =====================================================================

VARS_CORREL = ["TrackTemp_driver_mean", "AirTemp_mean", "Humidity_mean", "WindSpeed_mean",
               "ShareRainLaps", "AvgLapTimePct_vsCircuit_w", "LapTimePctStd_w",
               "ErrorRate_per100laps_w", "NumPenalties_driving", "NumDeletedLaps",
               "DNF_Accident_num", "PctTime_SafetyCarOrVSC", "GridPositionAdj"]


def correlacoes(pc):
    log("3.9 Matrizes de correlação...")
    ROTULOS.update({"ShareRainLaps": "Voltas do piloto com chuva (prop.)",
                    "AvgLapTimePct_vsCircuit_w": "Ritmo vs. circuito (%)",
                    "LapTimePctStd_w": "Inconsistência (p.p.)",
                    "ErrorRate_per100laps_w": "Erros por 100 voltas"})
    d = pc[VARS_CORREL].astype(float)
    nomes = [rot(v) for v in VARS_CORREL]
    for metodo, aba, num in [("pearson", "T11_correlacao_pearson", 11),
                             ("spearman", "T12_correlacao_spearman", 12)]:
        r = d.corr(method=metodo)
        # p-valores par a par, para marcar significância
        p = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
        for i in r.index:
            for j in r.columns:
                ok = d[[i, j]].dropna()
                if i != j and len(ok) > 3:
                    f = stats.pearsonr if metodo == "pearson" else stats.spearmanr
                    p.loc[i, j] = f(ok[i], ok[j])[1]
        txt = r.round(2).astype(str) + p.map(estrelas)
        np.fill_diagonal(txt.values, "1")
        txt.index, txt.columns = nomes, nomes
        txt = txt.reset_index().rename(columns={"index": "Variável"})
        registrar_tabela(aba, f"Tabela {num} — Matriz de correlação de "
                              f"{'Pearson' if metodo == 'pearson' else 'Spearman'}", txt,
                         "Base piloto-corrida (amostra principal). *** p<0,01; ** p<0,05; * p<0,10.")
        if metodo == "pearson":
            # triângulo inferior, sem a diagonal (1ª linha e última coluna vazias)
            rr = r.iloc[1:, :-1]
            mascara = np.triu(np.ones_like(rr, dtype=bool), k=1)
            cmap = sns.diverging_palette(250, 25, s=75, l=55, center="light", as_cmap=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(rr, mask=mascara, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True,
                        fmt=".2f", annot_kws={"size": 7}, linewidths=0.5, linecolor="white",
                        xticklabels=nomes[:-1], yticklabels=nomes[1:],
                        cbar_kws={"shrink": 0.7, "label": "Correlação"}, ax=ax)
            ax.set_title("Figura 8 — Matriz de correlação de Pearson (piloto-corrida)")
            ax.grid(False)
            plt.xticks(rotation=45, ha="right")
            salvar_figura(fig, "F08_correlacao", "mapa de calor das correlações")


# =====================================================================
# 3.10 RECORTE POR ERA DO REGULAMENTO (Tabela 13)
# =====================================================================

def por_era(pc, corr):
    log("3.10 Recorte por era do regulamento...")
    t = corr.groupby("Era").agg(Corridas=("RaceId", "nunique"),
                                Corridas_molhadas=("WetRace", "sum"),
                                Temp_pista_media=("TrackTemp_mean", "mean"))
    t2 = pc.groupby("Era").agg(Ritmo_vs_circuito=("AvgLapTimePct_vsCircuit_w", "mean"),
                               Inconsistencia=("LapTimePctStd_w", "mean"),
                               Erros_100_voltas=("ErrorRate_per100laps_w", "mean"),
                               Voltas_anuladas=("NumDeletedLaps", "mean"),
                               Prop_abandono=("DNF_num", "mean"),
                               Prop_abandono_acidente=("DNF_Accident_num", "mean"))
    t = t.join(t2).round(3).reset_index()
    registrar_tabela("T13_por_era", "Tabela 13 — Indicadores por era do regulamento técnico", t,
                     "Médias por observação piloto-corrida, exceto contagens de corridas. Diferenças "
                     "entre eras justificam o controle por equipe-temporada/temporada nos modelos.")


# =====================================================================
# 3.10b H3 — EXPERIÊNCIA DO PILOTO x EFEITO DA CHUVA (Tabela 14, Figura 9)
# =====================================================================
# Descritivo: a diferença entre corrida molhada e seca, em cada terço de
# experiência (GPs disputados antes da corrida). A H3 prevê que essa
# diferença seja MENOR entre os mais experientes. Sem controles: o teste
# formal está em 04_modelos.py.

def experiencia(pc):
    log("3.10b H3 — experiência x efeito da chuva...")
    d = pc.dropna(subset=["GPsAnteriores"]).copy()
    if d.empty:
        log("  experiência indisponível (rode 01_coleta.py --historico e 02_tratamento.py).")
        return
    d["PerdaControle"] = ((d["NumSpinsMsg"] > 0) | d["DNF_Accident"]).astype(int)
    d["VoltasLentas_100"] = np.where(d["NumLaps"] >= 10, 100 * d["NumSlowLaps"] / d["NumLaps"], np.nan)
    d["Terco"] = pd.qcut(d["GPsAnteriores"], 3, labels=["Menos experientes", "Intermediários",
                                                          "Mais experientes"])
    medidas = {"PerdaControle": "Rodada ou acidente (prop.)",
               "VoltasLentas_100": "Voltas lentas por 100 voltas",
               "AvgLapTimePct_vsRace_w": "Ritmo relativo ao grid (%)"}
    linhas = []
    for terco, g in d.groupby("Terco", observed=True):
        l = {"Terço de experiência": str(terco),
             "GPs anteriores (mín–máx)": f"{int(g['GPsAnteriores'].min())}–{int(g['GPsAnteriores'].max())}",
             "Idade média": g["Idade"].mean(), "N seca": int((~g["WetRace"]).sum()),
             "N molhada": int(g["WetRace"].sum())}
        for v, nome in medidas.items():
            seca, molh = g.loc[~g["WetRace"], v].dropna(), g.loc[g["WetRace"], v].dropna()
            l[f"{nome} — seca"] = seca.mean()
            l[f"{nome} — molhada"] = molh.mean()
            l[f"{nome} — diferença"] = molh.mean() - seca.mean()
        linhas.append(l)
    t14 = pd.DataFrame(linhas).round(3)
    r_idade = d[["GPsAnteriores", "Idade"]].corr().iloc[0, 1]
    registrar_tabela("T14_H3_experiencia", "Tabela 14 — Erros e ritmo em corridas secas e molhadas, "
                     "por terço de experiência do piloto (H3)", t14,
                     "Experiência = GPs em que o piloto largou antes da corrida, carreira inteira (API "
                     "Jolpica/Ergast). Terços com o mesmo número de observações. Diferença = molhada − seca. "
                     f"Correlação entre experiência e idade: {r_idade:.2f}. Comparação simples, sem controles "
                     "(ver Tabela M4 em 04_modelos.py).")
    log(f"  correlação experiência x idade = {r_idade:.2f}; "
        f"GPs anteriores: mediana {d['GPsAnteriores'].median():.0f}, máximo {d['GPsAnteriores'].max():.0f}")

    # Figura 9: diferença molhada − seca por terço (com IC 95%), duas medidas
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (v, nome) in zip(ax, list(medidas.items())[:2]):
        xs, ys, ics = [], [], []
        for i, (terco, g) in enumerate(d.groupby("Terco", observed=True)):
            seca, molh = g.loc[~g["WetRace"], v].dropna(), g.loc[g["WetRace"], v].dropna()
            dif = molh.mean() - seca.mean()
            ep = np.sqrt(molh.var() / len(molh) + seca.var() / len(seca)) if len(molh) > 1 else np.nan
            xs.append(i); ys.append(dif); ics.append(1.96 * ep)
        a.axhline(0, color=COR_NEUTRA, linewidth=1)
        a.errorbar(xs, ys, yerr=ics, fmt="o", color=COR_MOLHADO, ecolor=COR_MOLHADO,
                   elinewidth=1.5, markersize=7, capsize=0)
        a.set_xticks(xs, ["Menos\nexperientes", "Intermediários", "Mais\nexperientes"])
        a.set_xlim(-0.5, 2.5)
        a.set_title(nome, fontsize=10)
        a.set_ylabel("Molhada − seca")
        a.grid(axis="x", visible=False)
    fig.suptitle("Figura 9 — Efeito bruto da chuva por terço de experiência (IC 95%)", fontweight="bold")
    fig.tight_layout()
    salvar_figura(fig, "F09_H3_experiencia", "diferença molhada − seca por terço de experiência")


# =====================================================================
# 3.11 BASES FINAIS DE ANÁLISE (para a etapa 4)
# =====================================================================

def salvar_bases_finais(voltas, pc, corr, elegiveis):
    log("3.11 Salvando as bases finais de análise...")
    os.makedirs(FINAL_DIR, exist_ok=True)
    pc["Elegivel_H3"] = pc["Driver"].isin(elegiveis)
    pc["WetRace_num"] = pc["WetRace"].astype(int)
    voltas[voltas["IsPaceLap"]].to_csv(os.path.join(FINAL_DIR, "base_voltas_ritmo.csv"), index=False)
    pc.to_csv(os.path.join(FINAL_DIR, "base_piloto_corrida.csv"), index=False)
    corr.to_csv(os.path.join(FINAL_DIR, "base_corridas.csv"), index=False)
    log(f"  bases finais em {FINAL_DIR} (amostra principal; variáveis winsorizadas com sufixo _w)")


# =====================================================================
# EXECUÇÃO
# =====================================================================

def main():
    for d in (TAB_DIR, FIG_DIR, FINAL_DIR):
        os.makedirs(d, exist_ok=True)
    estilo_graficos()

    voltas, pc, corr = carregar()
    composicao_amostra(voltas, pc, corr)
    estatisticas_descritivas(voltas, pc, corr)
    outliers_e_winsorizacao(voltas, pc)
    elegiveis = viabilidade(voltas, pc, corr)
    hipotese_h1(pc)
    hipotese_h2(pc)
    hipotese_h3(pc, elegiveis)
    correlacoes(pc)
    por_era(pc, corr)
    experiencia(pc)
    salvar_bases_finais(voltas, pc, corr, elegiveis)
    gravar_excel()

    with open(os.path.join(RES_DIR, "resumo_exploracao.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(RESUMO))
    print("Exploração concluída. Resultados em", RES_DIR)


if __name__ == "__main__":
    main()