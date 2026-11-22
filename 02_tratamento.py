"""
PRÉ-REQUISITO
    Dados brutos em dados/brutos/ (gerados por src/01_coleta.py).
    Este código só LÊ os dados brutos — nunca os altera — e não acessa a
    internet.

O QUE FAZ
    2.1 Leitura e conversão dos arquivos brutos
    2.2 Base de voltas: clima de cada volta, voltas limpas/de ritmo, ritmo
        relativo à corrida e ao circuito, resíduo no stint
    2.3 Eventos da direção de prova: incidentes únicos, motivo limpo e
        classificado (pilotagem / procedimento / técnico / outros)
    2.4 Resultado oficial e classificação dos abandonos; sinais do momento
        do abandono e regras de aproximação de acidente
    2.5 Base piloto-corrida: desempenho, consistência e erros
    2.6 Base de corridas: clima, chuva e neutralizações
    2.7 Dicionário de variáveis
    2.8 Execução: validação da aproximação de acidente, medidas de erro,
        amostra principal, salvamento das bases e diagnósticos

    Os parâmetros (limites, regras, exclusões) ficam na seção 0 deste arquivo.

COMO RODAR (terminal do VS Code, na pasta raiz do projeto)
    python src/02_tratamento.py

SAÍDAS (dados/tratados/)
    voltas_tratadas.csv        -> 1 linha = 1 volta de 1 piloto      (H1, H3)
    piloto_corrida.csv         -> 1 linha = 1 piloto em 1 corrida   (H2, H3)
    corridas.csv               -> 1 linha = 1 corrida
    eventos_direcao_prova.csv  -> 1 linha = 1 evento envolvendo 1 piloto
    dicionario_variaveis.csv   -> definição e unidade das variáveis
    diagnostico/               -> conferências da qualidade do tratamento

PRÓXIMO PASSO
    python src/03_exploracao.py
"""

import json
import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =====================================================================
# 0.1 CAMINHOS
# =====================================================================

# Pasta raiz do projeto: a pasta acima de src/ (ou a própria pasta, se os
# códigos estiverem soltos na raiz). É onde ficam f1_cache/, dados/ e
# resultados/.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
DADOS_DIR = os.path.join(BASE_DIR, "dados")
BRUTOS_CORRIDAS_DIR = os.path.join(DADOS_DIR, "brutos", "corridas")   # entrada
HISTORICO_DIR = os.path.join(DADOS_DIR, "brutos", "historico_pilotos")  # entrada (H3)
TRATADOS_DIR = os.path.join(DADOS_DIR, "tratados")                    # saída
DIAGNOSTICO_DIR = os.path.join(TRATADOS_DIR, "diagnostico")

# Nomes dos arquivos brutos de cada corrida (gerados por 01_coleta.py)
ARQ_VOLTAS, ARQ_CLIMA = "voltas.csv", "clima.csv"
ARQ_RESULTADOS, ARQ_MENSAGENS = "resultados.csv", "mensagens_direcao_prova.csv"
ARQ_STATUS, ARQ_META = "status_pista.csv", "meta.json"


# =====================================================================
# 0.2 PARÂMETROS DO TRATAMENTO (decisões metodológicas)
# =====================================================================

# Volta "lenta suspeita": volta limpa mais de X% acima da mediana da corrida
# (no mesmo tipo de pneu: seco ou molhado). Pode indicar erro (rodada,
# escapada) ou dano. Não é excluída da base — apenas sinalizada.
LIMITE_VOLTA_LENTA_PCT = 10.0
# A referência da volta lenta é a mediana das voltas limpas de TODOS os
# pilotos numa janela de voltas ao redor (volta n ± JANELA_VOLTA_LENTA), no
# mesmo tipo de pneu. Assim, mudanças de condição ao longo da corrida (pista
# secando, chuva começando) não são confundidas com erro do piloto.
JANELA_VOLTA_LENTA = 2
# Leituras impossíveis do sensor de clima viram vazio (NaN)
TEMP_PISTA_MIN_VALIDA = 5.0     # °C (ex.: leituras de 0,0 °C)
TEMP_AR_MIN_VALIDA = 0.0        # °C
# Corrida molhada: >= 10% das voltas com pneu de chuva OU >= 10% do tempo
# com chuva registrada pela estação meteorológica.
LIMITE_CORRIDA_MOLHADA = 0.10
# Tolerância (s) para associar uma volta à medição de clima mais próxima
TOLERANCIA_CLIMA_S = 180
# Regra de classificação da FIA: piloto classificado se completou ao menos
# 90% das voltas do vencedor (usada quando o Status vem vazio).
LIMITE_CLASSIFICACAO = 0.90
# Regra usada para aproximar "abandono por acidente" quando a causa não é
# informada (2023–2026). "auto" = regra com maior F1 na validação contra a
# causa oficial. Para fixar uma regra, use o nome dela (ex.: "R2_volta1_ou_mensagem").
# Ver diagnostico/validacao_proxy_acidente.csv e validacao_proxy_por_condicao.csv.
REGRA_PROXY_ACIDENTE = "R3_sem_box_e_SC"
# Códigos de TrackStatus da F1: 1 verde, 2 amarela, 4 Safety Car,
# 5 bandeira vermelha, 6 VSC, 7 fim do VSC
CODIGOS_NEUTRALIZACAO = ("4", "5", "6", "7")
COMPOSTOS_MOLHADOS = ("INTERMEDIATE", "WET")

# Corridas fora da amostra principal (ficam nas bases, marcadas com
# AmostraPrincipal = False e o motivo da exclusão).
CORRIDAS_EXCLUIDAS = {
    "2021_12": "GP da Bélgica 2021: encerrado após poucas voltas atrás do "
               "Safety Car, sem disputa em bandeira verde",
}

# Eras do regulamento técnico (controle e recorte descritivo)
def era_regulamento(ano: int) -> str:
    if ano <= 2021:
        return "2021 (pré-efeito solo)"
    if ano <= 2025:
        return "2022–2025 (efeito solo)"
    return "2026 (novo regulamento)"


# =====================================================================
# 0.3 REGRAS DE CLASSIFICAÇÃO (resultado oficial e direção de prova)
# =====================================================================

# Classificação do "Status" do resultado final
STATUS_ACIDENTE = ("ACCIDENT", "COLLISION", "CRASH", "SPUN", "DAMAGE")
STATUS_GENERICO = ("RETIRED", "DNF", "NOT CLASSIFIED")

# Classificação do motivo dos eventos da direção de prova. A ordem de teste
# é: procedimento -> técnico -> pilotagem -> outros.
#   pilotagem    = erro de condução (relacionado à hipótese de erro)
#   procedimento = infração de regra esportiva (box, largada, SC, bandeiras)
#   tecnico      = problema do carro (não é erro do piloto)
MOTIVOS_PROCEDIMENTO = (
    "PIT LANE", "PIT ENTRY", "PIT EXIT", "PIT STOP", "SPEEDING", "RELEAS",
    "START", "FORMATION", "SAFETY CAR", "SC LINE", "VSC", "RED FLAG",
    "RED LIGHT", "YELLOW", "IMPEDING", "TYRE", "TECHNICAL", "PARC FERME",
    "FAILING TO", "OVERTAKING", "BLUE FLAG", "FUEL", "POWER UNIT", "GRID",
    "CROSSING", "INSTRUCTION", "DRS", "BEFORE SIGNAL", "JUMP START",
    "MAXIMUM DELTA", "OUT OF POSITION", "SERVING", "SERVE", "CAR LENGTHS",
    "UNNECESSARILY SLOWLY", "UNSPORTSMANLIKE", "NOT ENTERING", "NOT USING",
    "WHEELS NOT FITTED", "AFTER RECEIVING ASSISTANCE",
)
MOTIVOS_TECNICO = (
    "LOOSE", "UNSAFE CONDITION", "SAFETY LIGHT", "DRIVING IN AN UNSAFE",
)
MOTIVOS_PILOTAGEM = (
    "COLLISION", "LEAVING THE TRACK", "FORCING", "CHANGE OF DIRECTION",
    "TRACK LIMIT", "SPUN", "SPIN", "UNSAFE REJOIN", "REJOINING", "ERRATIC",
    "DANGEROUS", "MOVING UNDER BRAKING", "MOVEMENT UNDER BRAK", "OFF TRACK",
    "CONTACT", "LOSS OF CONTROL", "WEAVING",
)
# Textos que são DECISÃO dos comissários, e não motivo da investigação
DECISOES = (
    "NO FURTHER ACTION", "NO FURTHER INVESTIGATION", "NO INVESTIGATION",
    "WILL BE INVESTIGATED AFTER", "UNDER INVESTIGATION", "REVIEWED",
    "NOTED", "POSITIONS GIVEN BACK", "POSITION GIVEN BACK",
    "PLACE WAS GIVEN BACK", "NO ACTION",
)


# =====================================================================
# 0.4 FUNÇÕES AUXILIARES
# =====================================================================

def log(msg: str):
    """Imprime uma mensagem com horário (acompanha o andamento)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def id_corrida(meta: dict) -> str:
    """Identificador único da corrida: '<temporada>_<etapa>' (ex.: 2024_05)."""
    return f"{meta['Season']}_{int(meta['Round']):02d}"


def corrida_ja_coletada(pasta: str) -> bool:
    """A corrida só conta como coletada se o meta.json existir (ele é o
    último arquivo escrito, então uma coleta interrompida não é aceita)."""
    return os.path.exists(os.path.join(pasta, ARQ_META))




# =====================================================================
# ETAPA 2 — TRATAMENTO E CRIAÇÃO DAS VARIÁVEIS
# =====================================================================

# ---------- 2.1 Funções auxiliares de leitura e conversão ----------

COLUNAS_TEMPO_VOLTAS = [
    "Time", "LapTime", "PitOutTime", "PitInTime", "LapStartTime",
    "Sector1Time", "Sector2Time", "Sector3Time",
]


def ler_bruto(pasta: str, arquivo: str, **kw) -> pd.DataFrame:
    caminho = os.path.join(pasta, arquivo)
    if not os.path.exists(caminho) or os.path.getsize(caminho) < 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(caminho, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def para_segundos(serie: pd.Series) -> pd.Series:
    """Converte textos de tempo da FastF1 ('0 days 00:01:32.123') em segundos."""
    return pd.to_timedelta(serie, errors="coerce").dt.total_seconds()


def para_bool(serie: pd.Series) -> pd.Series:
    """Converte 'True'/'False'/vazio lidos do CSV em booleano (vazio=False)."""
    return serie.astype(str).str.strip().str.upper().isin(["TRUE", "1", "1.0"])


def listar_corridas_brutas() -> list:
    pastas = []
    if not os.path.isdir(BRUTOS_CORRIDAS_DIR):
        return pastas
    for ano in sorted(os.listdir(BRUTOS_CORRIDAS_DIR)):
        dir_ano = os.path.join(BRUTOS_CORRIDAS_DIR, ano)
        if not os.path.isdir(dir_ano):
            continue
        for p in sorted(os.listdir(dir_ano)):
            pasta = os.path.join(dir_ano, p)
            if os.path.isdir(pasta) and corrida_ja_coletada(pasta):
                pastas.append(pasta)
    return pastas


# ---------- 2.2 Tratamento da base de voltas ----------

def limpar_clima(w: pd.DataFrame) -> pd.DataFrame:
    """Converte o tempo para segundos, a chuva para 0/1 e descarta leituras
    impossíveis do sensor (ex.: temperatura da pista igual a 0 °C)."""
    w = w.copy()
    w["Time_s"] = para_segundos(w["Time"])
    w["Rainfall"] = para_bool(w["Rainfall"])
    for c in ("TrackTemp", "AirTemp", "Humidity", "WindSpeed", "Pressure"):
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    if "TrackTemp" in w.columns:
        w.loc[w["TrackTemp"] < TEMP_PISTA_MIN_VALIDA, "TrackTemp"] = np.nan
    if "AirTemp" in w.columns:
        w.loc[w["AirTemp"] < TEMP_AR_MIN_VALIDA, "AirTemp"] = np.nan
    return w


def mediana_janela(v: pd.DataFrame) -> pd.Series:
    """Mediana das voltas limpas de todos os pilotos nas voltas n−J … n+J,
    separada por tipo de pneu (seco x molhado). Se a janela tiver menos de
    5 voltas limpas, usa a mediana da corrida naquele tipo de pneu."""
    limpas = v[v["IsCleanLap"]]
    ref = pd.Series(np.nan, index=v.index)
    for molhado, g in v.groupby("IsWetCompound"):
        lg = limpas[limpas["IsWetCompound"] == molhado]
        if lg.empty:
            continue
        mediana_geral = lg["LapTimeSeconds"].median()
        tabela = {}
        for n in g["LapNumber"].dropna().unique():
            jan = lg.loc[(lg["LapNumber"] >= n - JANELA_VOLTA_LENTA)
                         & (lg["LapNumber"] <= n + JANELA_VOLTA_LENTA), "LapTimeSeconds"]
            tabela[n] = jan.median() if len(jan) >= 5 else mediana_geral
        ref.loc[g.index] = g["LapNumber"].map(tabela).astype(float)
    return ref


def tratar_voltas(voltas: pd.DataFrame, clima: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Limpa as voltas, associa o clima de cada volta e cria indicadores."""
    v = voltas.copy()

    # (a) tempos em segundos
    for c in COLUNAS_TEMPO_VOLTAS:
        if c in v.columns:
            v[c + "_s"] = para_segundos(v[c])
    v = v.rename(columns={"LapTime_s": "LapTimeSeconds"})

    # (b) remove linhas sem identificação e voltas duplicadas
    v = v.dropna(subset=["Driver", "LapNumber"])
    v = v.drop_duplicates(subset=["Driver", "LapNumber"], keep="first")

    # (c) clima da volta: medição mais próxima do MEIO da volta
    #     (a estação meteorológica da F1 registra cerca de 1 vez por minuto)
    v["LapMid_s"] = v["LapStartTime_s"] + v["LapTimeSeconds"] / 2
    v["LapMid_s"] = v["LapMid_s"].fillna(v["Time_s"])
    col_clima = ["AirTemp", "TrackTemp", "Humidity", "Pressure",
                 "Rainfall", "WindSpeed", "WindDirection"]
    if not clima.empty:
        w = limpar_clima(clima)
        w = w.dropna(subset=["Time_s"]).sort_values("Time_s")
        w = w[["Time_s"] + [c for c in col_clima if c in w.columns]]
        w = w.rename(columns={"Time_s": "LapMid_s"})
        com_chave = v[v["LapMid_s"].notna()].sort_values("LapMid_s")
        sem_chave = v[v["LapMid_s"].isna()]
        com_chave = pd.merge_asof(com_chave, w, on="LapMid_s",
                                  direction="nearest",
                                  tolerance=TOLERANCIA_CLIMA_S)
        v = pd.concat([com_chave, sem_chave], ignore_index=True)
    for c in col_clima:
        if c not in v.columns:
            v[c] = np.nan
    v["Rainfall"] = v["Rainfall"].astype("boolean")

    # (d) indicadores de cada volta
    ts = v["TrackStatus"].astype(str).str.replace(r"\.0$", "", regex=True)
    v["TrackStatus"] = ts
    v["IsNeutralized"] = ts.apply(lambda s: any(c in s for c in CODIGOS_NEUTRALIZACAO))
    v["IsYellow"] = ts.str.contains("2", na=False)
    v["IsPitInLap"] = v["PitInTime_s"].notna()
    v["IsPitOutLap"] = v["PitOutTime_s"].notna()
    v["IsPitLap"] = v["IsPitInLap"] | v["IsPitOutLap"]
    v["IsLap1"] = v["LapNumber"] == 1
    v["Compound"] = v["Compound"].astype(str).str.upper().replace({"NAN": np.nan, "NONE": np.nan})
    v["IsWetCompound"] = v["Compound"].isin(COMPOSTOS_MOLHADOS)
    for c in ("Deleted", "FastF1Generated", "IsAccurate", "FreshTyre"):
        v[c] = para_bool(v[c]) if c in v.columns else False

    # Volta limpa (estrutural): tem tempo, não é a 1ª volta, não é volta de
    # box, não ocorreu sob SC/VSC/bandeira vermelha e não foi gerada
    # artificialmente pela FastF1.
    v["IsCleanLap"] = (v["LapTimeSeconds"].notna() & ~v["IsLap1"] & ~v["IsPitLap"]
                       & ~v["IsNeutralized"] & ~v["FastF1Generated"])

    # (e) ritmo relativo à corrida: % acima da mediana das voltas limpas da
    #     mesma corrida e da mesma condição de pneu (seco x molhado).
    med = (v[v["IsCleanLap"]].groupby("IsWetCompound")["LapTimeSeconds"]
           .median().rename("RaceMedianLap"))
    v = v.merge(med, left_on="IsWetCompound", right_index=True, how="left")
    v["LapTimePct_vsRace"] = 100 * (v["LapTimeSeconds"] / v["RaceMedianLap"] - 1)
    # Volta lenta suspeita: comparada às voltas dos outros pilotos no mesmo
    # momento da corrida (janela de voltas), não à mediana da corrida inteira.
    v["WindowMedianLap"] = mediana_janela(v)
    v["LapTimePct_vsWindow"] = 100 * (v["LapTimeSeconds"] / v["WindowMedianLap"] - 1)
    v["IsSlowLap"] = v["IsCleanLap"] & (v["LapTimePct_vsWindow"] > LIMITE_VOLTA_LENTA_PCT)
    # Volta de ritmo: volta limpa que não é lenta suspeita. Usada nas
    # medidas de desempenho médio e de consistência.
    v["IsPaceLap"] = v["IsCleanLap"] & ~v["IsSlowLap"]

    # (f) identificação da corrida
    v["Season"] = meta["Season"]
    v["Round"] = meta["Round"]
    v["EventName"] = meta["EventName"]
    v["Location"] = meta.get("Location")
    v["RaceId"] = id_corrida(meta)
    return v.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def residuo_no_stint(df: pd.DataFrame) -> pd.Series:
    """Resíduo do ritmo dentro de cada stint, depois de retirar a tendência
    linear de desgaste do pneu (TyreLife). Mede a irregularidade do piloto
    que não é explicada pela degradação natural do pneu e pelo consumo de
    combustível (que caminham juntos dentro do stint)."""
    y = df["LapTimePct_vsRace"].to_numpy(dtype=float)
    x = df["TyreLife"].to_numpy(dtype=float)
    ok = ~np.isnan(y) & ~np.isnan(x)
    res = pd.Series(np.nan, index=df.index)
    if ok.sum() >= 4 and np.nanstd(x[ok]) > 0:
        b, a = np.polyfit(x[ok], y[ok], 1)
        res[ok] = y[ok] - (a + b * x[ok])
    elif ok.sum() >= 2:
        res[ok] = y[ok] - y[ok].mean()
    return res


# ---------- 2.3 Eventos da direção de prova (erros por piloto) ----------

REGEX_CARRO = re.compile(r"\b(\d{1,2})\s*\(([A-Z]{3})\)")
REGEX_LIMPEZA_MOTIVO = [
    re.compile(r"\(\s*\d{1,2}:\d{1,2}(:+\d{1,2})?\s*\)"),       # (16:11:31)
    re.compile(r"\(\s*\d+\s*(ST|ND|RD|TH)\s+OFFENCE\s*\)"),      # (5TH OFFENCE)
    re.compile(r"\bAT\s+(TURN|T)\s*\d+\b"),                       # AT TURN 4 / AT T8
    re.compile(r"\bLAP\s+\d+\b"),                                 # LAP 5
    re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b"),                       # 14:12:10
]


def tipo_mensagem(msg: str):
    """Classifica a mensagem da direção de prova em um tipo de evento."""
    if "SERVED" in msg:                       # cumprimento de pena: ignora
        return None
    if "DELETED" in msg and "TRACK LIMITS" in msg:
        return "volta_deletada"
    if "BLACK AND WHITE" in msg:
        return "bandeira_preta_branca"
    if "REPRIMAND" in msg:
        return "advertencia"
    if "PENALTY" in msg and ("STEWARDS" in msg or "FOR CAR" in msg):
        return "penalidade"
    if "INCIDENT" in msg or "INVESTIGAT" in msg:
        return "incidente"
    if "SPUN" in msg or "SPIN" in msg:
        return "rodada"
    return None


def limpar_motivo(msg: str):
    """Separa o MOTIVO (ex.: 'CAUSING A COLLISION') da DECISÃO (ex.: 'NOTED',
    'NO FURTHER ACTION'), removendo horários, curva e nº da infração."""
    bruto = msg.rsplit(" - ", 1)[1].strip() if " - " in msg else ""
    decisao = next((d for d in DECISOES if d in msg), "")
    # texto após o hífen que é decisão, ou repetição da própria mensagem
    if not bruto or any(bruto.startswith(d) for d in DECISOES) or "INCIDENT" in bruto:
        bruto = ""
    for rx in REGEX_LIMPEZA_MOTIVO:
        bruto = rx.sub("", bruto)
    bruto = re.sub(r"\s+", " ", bruto).strip(" ,.-–")
    return bruto, decisao


def classe_motivo(tipo: str, motivo: str) -> str:
    if tipo in ("volta_deletada", "rodada"):
        return "pilotagem"
    if any(k in motivo for k in MOTIVOS_PROCEDIMENTO):
        return "procedimento"
    if any(k in motivo for k in MOTIVOS_TECNICO):
        return "tecnico"
    if any(k in motivo for k in MOTIVOS_PILOTAGEM):
        return "pilotagem"
    if tipo == "bandeira_preta_branca":   # advertência por conduta na pista
        return "pilotagem"
    return "outros"


def local_incidente(msg: str) -> str:
    """Curva do incidente (ex.: 'TURN 12'); usada para unir as várias
    mensagens de um mesmo incidente."""
    m = re.search(r"TURN\s*\d+", msg)
    return m.group(0).replace(" ", "") if m else ""


def tratar_mensagens(mensagens: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Transforma as mensagens brutas em eventos únicos por piloto.

    Incidentes: as mensagens de uma mesma ocorrência ('NOTED', 'UNDER
    INVESTIGATION', 'NO FURTHER ACTION'...) são unidas pela chave
    (curva, carros envolvidos); o motivo é o primeiro informado em qualquer
    uma delas e a decisão é a última. Os demais tipos (volta deletada,
    penalidade etc.) geram um evento por mensagem distinta."""
    colunas = ["RaceId", "Season", "Round", "EventName", "Lap", "Time_utc",
               "Driver", "DriverNumber", "TipoEvento", "Motivo",
               "ClasseMotivo", "NumCarrosEnvolvidos", "Decisao", "Mensagem"]
    if mensagens.empty or "Message" not in mensagens.columns:
        return pd.DataFrame(columns=colunas)

    m = mensagens.copy()
    m["Message"] = m["Message"].fillna("").astype(str).str.upper()
    m = m.drop_duplicates(subset=["Message"])
    if "Time" in m.columns:
        m = m.sort_values("Time")

    eventos = {}      # chave -> dicionário do evento
    for _, r in m.iterrows():
        msg = r["Message"]
        tipo = tipo_mensagem(msg)
        if tipo is None:
            continue
        carros = tuple(sorted(set(REGEX_CARRO.findall(msg))))
        if not carros:
            continue
        motivo, decisao = limpar_motivo(msg)
        if tipo == "volta_deletada":
            motivo = "TRACK LIMITS"
        chave = ("incidente", local_incidente(msg), carros) if tipo == "incidente" \
            else (tipo, msg)
        if chave in eventos:                       # mesma ocorrência
            ev = eventos[chave]
            if not ev["Motivo"] and motivo:
                ev["Motivo"] = motivo
            if decisao:
                ev["Decisao"] = decisao
            continue
        eventos[chave] = {"Tipo": tipo, "Carros": carros, "Motivo": motivo,
                          "Decisao": decisao, "Lap": r.get("Lap", np.nan),
                          "Time": r.get("Time", np.nan), "Mensagem": msg}

    linhas = []
    for ev in eventos.values():
        classe = classe_motivo(ev["Tipo"], ev["Motivo"])
        for numero, sigla in ev["Carros"]:
            linhas.append({
                "RaceId": id_corrida(meta), "Season": meta["Season"],
                "Round": meta["Round"], "EventName": meta["EventName"],
                "Lap": ev["Lap"], "Time_utc": ev["Time"],
                "Driver": sigla, "DriverNumber": numero,
                "TipoEvento": ev["Tipo"], "Motivo": ev["Motivo"],
                "ClasseMotivo": classe,
                "NumCarrosEnvolvidos": len(ev["Carros"]),
                "Decisao": ev["Decisao"], "Mensagem": ev["Mensagem"],
            })
    return pd.DataFrame(linhas, columns=colunas)


# ---------- 2.4 Resultado final e classificação de abandonos ----------

def classificar_status(status, posicao_classificada, ultima_volta, voltas_lider) -> str:
    """Classifica o Status oficial do resultado.
    Retorna: Finished, Accident_or_error, Mechanical_or_other,
             Retired_generic (abandono sem causa informada), DNS, DSQ.
    Se o Status vier vazio, usa a posição classificada e, em último caso,
    a regra da FIA (completou >= 90% das voltas do vencedor)."""
    s = "" if pd.isna(status) else str(status).upper().strip()
    pc = "" if pd.isna(posicao_classificada) else str(posicao_classificada).upper().strip()
    if s:
        if "FINISHED" in s or s.startswith("+") or "LAP" in s:
            return "Finished"
        if "DISQUALIFIED" in s or "EXCLUDED" in s:
            return "DSQ"
        if "DID NOT START" in s or "WITHDREW" in s:
            return "DNS"
        if any(k in s for k in STATUS_ACIDENTE):
            return "Accident_or_error"
        if s in STATUS_GENERICO:
            return "Retired_generic"
        return "Mechanical_or_other"
    # Status vazio
    if pc in ("D", "E"):
        return "DSQ"
    if pc == "W":
        return "DNS"
    if pc.replace(".0", "").isdigit():
        return "Finished"
    if pd.isna(ultima_volta) or ultima_volta == 0:
        return "DNS"
    if voltas_lider and ultima_volta >= LIMITE_CLASSIFICACAO * voltas_lider:
        return "Finished"
    return "Retired_generic"


def sinais_abandono(v_piloto: pd.DataFrame, status: pd.DataFrame,
                    eventos_piloto: pd.DataFrame, volta_mediana: float) -> dict:
    """Sinais observáveis no momento do abandono, usados para aproximar a
    causa quando a fonte não a informa:
      - Abandono_Volta1: abandonou na 1ª volta (quase sempre batida);
      - Recolheu_Box: a última volta termina com entrada no box (típico de
        quebra: o carro é levado à garagem);
      - Neutralizacao_SC: Safety Car, VSC ou bandeira vermelha acionados
        logo após o início da última volta (carro parado na pista);
      - Neutralizacao_Amarela: idem, incluindo bandeira amarela;
      - Mensagem_Pilotagem: incidente/rodada/penalidade de pilotagem
        envolvendo o piloto na última volta ou na anterior."""
    ult = v_piloto.sort_values("LapNumber").iloc[-1]
    ultima_volta = ult["LapNumber"]
    inicio = ult.get("LapStartTime_s", np.nan)
    fim_janela = inicio + 2.0 * (volta_mediana if pd.notna(volta_mediana) else 100)

    sc = amarela = False
    if not status.empty and pd.notna(inicio):
        janela = status[(status["Time_s"] >= inicio - 5) & (status["Time_s"] <= fim_janela)]
        sc = janela["Status"].isin(["4", "5", "6"]).any()
        amarela = janela["Status"].isin(["2", "4", "5", "6"]).any()

    msg = False
    if not eventos_piloto.empty:
        e = eventos_piloto[(eventos_piloto["ClasseMotivo"] == "pilotagem")
                           & eventos_piloto["TipoEvento"].isin(["incidente", "rodada", "penalidade"])]
        lap = pd.to_numeric(e["Lap"], errors="coerce")
        msg = bool(((lap >= ultima_volta - 1) & (lap <= ultima_volta + 1)).any())

    return {"Abandono_Volta1": bool(ultima_volta <= 1),
            "Recolheu_Box": bool(pd.notna(ult.get("PitInTime_s", np.nan))),
            "Neutralizacao_SC": bool(sc), "Neutralizacao_Amarela": bool(amarela),
            "Mensagem_Pilotagem": msg}


# Regras candidatas para aproximar "abandono por acidente" quando a causa
# não é informada. Cada regra é avaliada contra a causa oficial (2021–2022).
REGRAS_PROXY = {
    "R1_mensagem": lambda s: s["Mensagem_Pilotagem"],
    "R2_volta1_ou_mensagem": lambda s: s["Abandono_Volta1"] | s["Mensagem_Pilotagem"],
    "R3_sem_box_e_SC": lambda s: ~s["Recolheu_Box"] & (
        s["Abandono_Volta1"] | s["Mensagem_Pilotagem"] | s["Neutralizacao_SC"]),
    "R4_sem_box_e_amarela": lambda s: ~s["Recolheu_Box"] & (
        s["Abandono_Volta1"] | s["Mensagem_Pilotagem"] | s["Neutralizacao_Amarela"]),
    "R5_sem_box": lambda s: ~s["Recolheu_Box"],
}


# ---------- 2.5 Montagem da base piloto-corrida ----------

def montar_piloto_corrida(v: pd.DataFrame, resultados: pd.DataFrame,
                          eventos: pd.DataFrame, status: pd.DataFrame,
                          meta: dict) -> pd.DataFrame:
    ritmo = v[v["IsPaceLap"]]

    # (a) desempenho e consistência (somente voltas de ritmo)
    agg = ritmo.groupby("Driver").agg(
        NumPaceLaps=("LapNumber", "count"),
        AvgLapTimePct_vsRace=("LapTimePct_vsRace", "mean"),
        MedianLapTimePct_vsRace=("LapTimePct_vsRace", "median"),
        LapTimePctStd=("LapTimePct_vsRace", "std"),
        ConsistencyResidStd=("StintResidual", "std"),
        AvgLapTimePct_vsCircuit=("LapTimePct_vsCircuit", "mean"),
    )
    # (b) exposição e clima vividos pelo piloto (todas as voltas)
    # Chuva como número (1/0/vazio) ANTES de agregar: agregar a coluna
    # booleana devolvia True/False em vez da proporção.
    v = v.assign(Rainfall_num=pd.to_numeric(v["Rainfall"].astype("Float64"), errors="coerce"))
    expo = v.groupby("Driver").agg(
        Team=("Team", "first"),
        DriverNumber=("DriverNumber", "first"),
        NumLaps=("LapNumber", "count"),
        LastLap=("LapNumber", "max"),
        NumPitStops=("IsPitInLap", "sum"),
        NumDeletedLaps=("Deleted", "sum"),
        NumSlowLaps=("IsSlowLap", "sum"),
        ShareWetCompoundLaps=("IsWetCompound", "mean"),
        ShareRainLaps=("Rainfall_num", "mean"),
        AirTemp_driver_mean=("AirTemp", "mean"),
        TrackTemp_driver_mean=("TrackTemp", "mean"),
    )
    pc = expo.join(agg, how="left").reset_index()

    # (c) eventos da direção de prova por piloto
    contagens = {
        "NumIncidents": lambda e: e["TipoEvento"].eq("incidente"),
        "NumIncidents_driving": lambda e: e["TipoEvento"].eq("incidente") & e["ClasseMotivo"].eq("pilotagem"),
        "NumPenalties": lambda e: e["TipoEvento"].eq("penalidade"),
        "NumPenalties_driving": lambda e: e["TipoEvento"].eq("penalidade") & e["ClasseMotivo"].eq("pilotagem"),
        "NumReprimands": lambda e: e["TipoEvento"].eq("advertencia"),
        "NumBlackWhiteFlags": lambda e: e["TipoEvento"].eq("bandeira_preta_branca"),
        "NumSpinsMsg": lambda e: e["TipoEvento"].eq("rodada"),
    }
    if not eventos.empty:
        ev = pd.DataFrame({k: f(eventos) for k, f in contagens.items()})
        ev["Driver"] = eventos["Driver"].values
        ev = ev.groupby("Driver").sum()
        pc = pc.merge(ev, left_on="Driver", right_index=True, how="left")
    for c in contagens:
        if c not in pc.columns:
            pc[c] = 0
        pc[c] = pc[c].fillna(0).astype(int)

    # (d) resultado oficial
    if not resultados.empty:
        r = resultados.rename(columns={"Abbreviation": "Driver"})
        cols = [c for c in ["Driver", "DriverId", "TeamName", "FullName", "GridPosition",
                            "Position", "ClassifiedPosition", "Status", "Points"]
                if c in r.columns]
        pc = pc.merge(r[cols], on="Driver", how="left")
    for c in ["GridPosition", "Position", "ClassifiedPosition", "Status", "Points"]:
        if c not in pc.columns:
            pc[c] = np.nan

    voltas_lider = v["LapNumber"].max()
    n_pilotos = pc["Driver"].nunique()
    pc["PitLaneStart"] = pc["GridPosition"].isna() | (pc["GridPosition"] == 0)
    pc["GridPositionAdj"] = pc["GridPosition"].where(~pc["PitLaneStart"], n_pilotos)
    pc["DNF_Category"] = [classificar_status(s, p, u, voltas_lider) for s, p, u in
                          zip(pc["Status"], pc["ClassifiedPosition"], pc["LastLap"])]
    pc["Finished"] = pc["DNF_Category"].eq("Finished")
    pc["DNF"] = pc["DNF_Category"].isin(["Accident_or_error", "Mechanical_or_other",
                                         "Retired_generic"])
    pc["PositionsGained"] = np.where(pc["Finished"],
                                     pc["GridPositionAdj"] - pc["Position"], np.nan)

    # (e) sinais do momento do abandono (para a aproximação de acidente)
    st = status.copy()
    if not st.empty:
        st["Time_s"] = para_segundos(st["Time"])
        st["Status"] = st["Status"].astype(str).str.replace(r"\.0$", "", regex=True)
    volta_mediana = v.loc[v["IsCleanLap"], "LapTimeSeconds"].median()
    sinais = []
    for d, dnf in zip(pc["Driver"], pc["DNF"]):
        if dnf:
            e_p = eventos[eventos["Driver"] == d] if not eventos.empty else eventos
            sinais.append(sinais_abandono(v[v["Driver"] == d], st, e_p, volta_mediana))
        else:
            sinais.append({k: False for k in ("Abandono_Volta1", "Recolheu_Box",
                                               "Neutralizacao_SC", "Neutralizacao_Amarela",
                                               "Mensagem_Pilotagem")})
    pc = pd.concat([pc, pd.DataFrame(sinais, index=pc.index)], axis=1)

    pc["Season"], pc["Round"] = meta["Season"], meta["Round"]
    pc["EventName"], pc["Location"] = meta["EventName"], meta.get("Location")
    pc["RaceId"] = id_corrida(meta)
    return pc


def avaliar_regras_proxy(pc: pd.DataFrame) -> tuple:
    """Compara cada regra candidata com a causa oficial do abandono, nas
    temporadas em que ela existe. Retorna a tabela de desempenho e o nome da
    regra com maior F1 (equilíbrio entre precisão e cobertura)."""
    base = pc[pc["DNF_Category"].isin(["Accident_or_error", "Mechanical_or_other"])]
    linhas = []
    if base.empty:
        return pd.DataFrame(), "R1_mensagem"
    real = base["DNF_Category"].eq("Accident_or_error")
    for nome, regra in REGRAS_PROXY.items():
        prev = regra(base).astype(bool)
        vp = int((prev & real).sum()); fp = int((prev & ~real).sum())
        fn = int((~prev & real).sum()); vn = int((~prev & ~real).sum())
        prec = vp / (vp + fp) if vp + fp else np.nan
        rec = vp / (vp + fn) if vp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
        linhas.append({"Regra": nome, "Verdadeiro_positivo": vp, "Falso_positivo": fp,
                       "Falso_negativo": fn, "Verdadeiro_negativo": vn,
                       "Precisao": prec, "Cobertura_recall": rec,
                       "Acuracia": (vp + vn) / len(base), "F1": f1,
                       "N_abandonos_com_causa": len(base),
                       "Temporadas": ", ".join(map(str, sorted(base["Season"].unique())))})
    tab = pd.DataFrame(linhas).sort_values("F1", ascending=False)
    return tab, tab.iloc[0]["Regra"]


def validar_proxy_por_condicao(pc: pd.DataFrame) -> pd.DataFrame:
    """Desempenho de cada regra separado entre corridas secas e molhadas.
    Importante para H2: se a regra erra mais na chuva (por exemplo, porque o
    Safety Car é mais frequente com chuva), o erro de medida fica correlacionado
    com a variável explicativa e pode distorcer o efeito estimado."""
    base = pc[pc["DNF_Category"].isin(["Accident_or_error", "Mechanical_or_other"])]
    linhas = []
    for (cond, g) in base.groupby(base["WetRace"].map({True: "Molhada", False: "Seca"})):
        real = g["DNF_Category"].eq("Accident_or_error")
        for nome, regra in REGRAS_PROXY.items():
            prev = regra(g).astype(bool)
            vp = int((prev & real).sum()); fp = int((prev & ~real).sum())
            fn = int((~prev & real).sum()); vn = int((~prev & ~real).sum())
            linhas.append({"Regra": nome, "Condicao": cond, "N_abandonos": len(g),
                           "Acidentes_reais": int(real.sum()),
                           "Previstos_como_acidente": int(prev.sum()),
                           "Precisao": vp / (vp + fp) if vp + fp else np.nan,
                           "Cobertura_recall": vp / (vp + fn) if vp + fn else np.nan,
                           "Taxa_falso_positivo": fp / (fp + vn) if fp + vn else np.nan,
                           "Vies_contagem": int(prev.sum()) - int(real.sum())})
    return pd.DataFrame(linhas).sort_values(["Regra", "Condicao"])


# ---------- 2.6 Montagem da base de corridas ----------

def resumo_status_pista(status: pd.DataFrame, inicio: float, fim: float) -> dict:
    """Percentual do tempo de corrida sob amarela, SC/VSC e bandeira vermelha."""
    vazio = {"PctTime_YellowOrWorse": np.nan, "PctTime_SafetyCarOrVSC": np.nan,
             "PctTime_RedFlag": np.nan, "NumTrackStatusEvents": 0}
    if status.empty or pd.isna(inicio) or pd.isna(fim) or fim <= inicio:
        return vazio
    ts = status.copy()
    ts["Time_s"] = para_segundos(ts["Time"])
    ts["Status"] = ts["Status"].astype(str).str.replace(r"\.0$", "", regex=True)
    ts = ts.dropna(subset=["Time_s"]).sort_values("Time_s").reset_index(drop=True)
    ts["Fim_s"] = ts["Time_s"].shift(-1).fillna(fim)       # último trecho vai até o fim
    ini = ts["Time_s"].clip(lower=inicio, upper=fim)
    fi = ts["Fim_s"].clip(lower=inicio, upper=fim)
    ts["Dur"] = (fi - ini).clip(lower=0)
    total = fim - inicio
    pct = lambda cods: ts.loc[ts["Status"].isin(cods), "Dur"].sum() / total
    dentro = ts[(ts["Time_s"] >= inicio) & (ts["Time_s"] <= fim)]
    return {"PctTime_YellowOrWorse": pct(["2", "4", "5", "6", "7"]),
            "PctTime_SafetyCarOrVSC": pct(["4", "6", "7"]),
            "PctTime_RedFlag": pct(["5"]),
            "NumTrackStatusEvents": int((dentro["Status"] != "1").sum())}


def montar_corrida(v, clima, status, eventos, meta) -> dict:
    inicio, fim = v["LapStartTime_s"].min(), v["Time_s"].max()
    linha = {"RaceId": id_corrida(meta),
             "Season": meta["Season"], "Round": meta["Round"],
             "EventName": meta["EventName"], "Location": meta.get("Location"),
             "Country": meta.get("Country"), "EventDate": meta.get("EventDate"),
             "NumDrivers": v["Driver"].nunique(),
             "NumLapsLeader": v["LapNumber"].max()}

    # Clima somente durante a corrida (exclui o período antes da largada)
    if not clima.empty:
        w = limpar_clima(clima)
        w = w[(w["Time_s"] >= inicio) & (w["Time_s"] <= fim)]
        linha.update({
            "AirTemp_mean": w["AirTemp"].mean(), "AirTemp_max": w["AirTemp"].max(),
            "TrackTemp_mean": w["TrackTemp"].mean(), "TrackTemp_max": w["TrackTemp"].max(),
            "TrackTemp_min": w["TrackTemp"].min(),
            "Humidity_mean": w["Humidity"].mean(), "WindSpeed_mean": w["WindSpeed"].mean(),
            "Rainfall_any": bool(w["Rainfall"].any()),
            "Rainfall_share_of_race": w["Rainfall"].mean(),
        })
    linha["ShareWetCompoundLaps"] = v["IsWetCompound"].mean()
    linha["WetRace"] = bool(linha["ShareWetCompoundLaps"] >= LIMITE_CORRIDA_MOLHADA
                            or (linha.get("Rainfall_share_of_race") or 0) >= LIMITE_CORRIDA_MOLHADA)
    linha.update(resumo_status_pista(status, inicio, fim))
    ev_inc = eventos[eventos["TipoEvento"] == "incidente"] if not eventos.empty else eventos
    linha["NumUniqueIncidents"] = int(ev_inc["Mensagem"].nunique()) if not eventos.empty else 0
    return linha


# ---------- 2.7 Dicionário de variáveis (apoio à tabela da metodologia) ----------

DICIONARIO = [
    # base, variável, descrição, unidade
    ("voltas", "LapTimeSeconds", "Tempo da volta", "segundos"),
    ("voltas", "LapTimePct_vsRace", "Diferença % do tempo da volta para a mediana das voltas limpas da corrida (mesma condição de pneu)", "%"),
    ("voltas", "LapTimePct_vsCircuit", "Diferença % para a mediana das voltas de ritmo em pista seca no mesmo circuito, 2021–2026", "%"),
    ("voltas", "StintResidual", "Resíduo do ritmo no stint após retirar a tendência de desgaste do pneu", "p.p."),
    ("voltas", "AirTemp / TrackTemp", "Temperatura do ar / do asfalto na medição mais próxima do meio da volta", "°C"),
    ("voltas", "Rainfall", "Chuva registrada pela estação no momento da volta", "0/1"),
    ("voltas", "IsWetCompound", "Volta com pneu intermediário ou de chuva", "0/1"),
    ("voltas", "IsCleanLap", "Volta sem box, sem 1ª volta e sem SC/VSC/bandeira vermelha", "0/1"),
    ("voltas", "IsSlowLap", f"Volta limpa mais de {LIMITE_VOLTA_LENTA_PCT:.0f}% acima da mediana dos outros pilotos nas voltas n±{JANELA_VOLTA_LENTA}, mesmo tipo de pneu (possível erro)", "0/1"),
    ("voltas", "IsPaceLap", "Volta limpa e não lenta; usada nas medidas de ritmo", "0/1"),
    ("voltas", "Deleted", "Volta anulada pela direção de prova (limite de pista)", "0/1"),
    ("voltas", "TyreLife / Compound / Stint", "Idade do pneu (voltas), composto e número do stint", "—"),
    ("piloto_corrida", "AvgLapTimePct_vsRace", "Média do ritmo relativo nas voltas de ritmo (desempenho relativo ao grid)", "%"),
    ("piloto_corrida", "AvgLapTimePct_vsCircuit", "Média do ritmo em relação à referência seca do circuito (desempenho absoluto)", "%"),
    ("piloto_corrida", "LapTimePctStd", "Desvio-padrão do ritmo relativo (inconsistência)", "p.p."),
    ("piloto_corrida", "ConsistencyResidStd", "Desvio-padrão do resíduo no stint (inconsistência sem efeito do pneu)", "p.p."),
    ("piloto_corrida", "NumDeletedLaps", "Voltas anuladas por limite de pista", "contagem"),
    ("piloto_corrida", "NumIncidents(_driving)", "Incidentes investigados envolvendo o piloto (todos / de pilotagem), contados uma vez", "contagem"),
    ("piloto_corrida", "NumPenalties(_driving)", "Penalidades recebidas (todas / por erro de pilotagem)", "contagem"),
    ("piloto_corrida", "NumSpinsMsg / NumSlowLaps", "Rodadas informadas pela direção de prova / voltas lentas suspeitas", "contagem"),
    ("piloto_corrida", "ErrorCount", "MEDIDA PRINCIPAL DE ERRO: voltas anuladas + penalidades de pilotagem + rodadas + voltas lentas", "contagem"),
    ("piloto_corrida", "ErrorRate_per100laps", "ErrorCount por 100 voltas disputadas", "por 100 voltas"),
    ("piloto_corrida", "AnyDrivingPenalty", "Recebeu ao menos uma penalidade por erro de pilotagem", "0/1"),
    ("piloto_corrida", "DNF", "Abandonou a corrida (exclui não largou e desclassificado)", "0/1"),
    ("piloto_corrida", "DNF_Accident", "Abandono por acidente: causa oficial (2021–2022) ou aproximação validada (2023–2026)", "0/1"),
    ("piloto_corrida", "DNF_Accident_Source", "Origem da informação de DNF_Accident: oficial ou aproximação", "texto"),
    ("piloto_corrida", "AnySeriousError", "Abandono por acidente ou penalidade de pilotagem", "0/1"),
    ("piloto_corrida", "ShareRainLaps / ShareWetCompoundLaps", "Proporção das voltas do piloto com chuva / pneu de chuva", "0–1"),
    ("piloto_corrida", "GridPositionAdj", "Posição de largada (largada do box = último lugar)", "posição"),
    ("todas", "Era", "Era do regulamento técnico: 2021; 2022–2025 (efeito solo); 2026 (novo regulamento)", "categoria"),
    ("todas", "TeamSeason", "Equipe × temporada (controle do desempenho do carro)", "categoria"),
    ("todas", "AmostraPrincipal", "Observação pertence à amostra principal (ver MotivoExclusao)", "0/1"),
    ("corridas", "AirTemp_mean / TrackTemp_mean", "Temperatura média durante a corrida", "°C"),
    ("corridas", "Rainfall_share_of_race", "Proporção das medições com chuva durante a corrida", "0–1"),
    ("corridas", "WetRace", f"Corrida com ≥{LIMITE_CORRIDA_MOLHADA:.0%} das voltas com pneu de chuva ou ≥{LIMITE_CORRIDA_MOLHADA:.0%} do tempo com chuva", "0/1"),
    ("corridas", "PctTime_SafetyCarOrVSC", "Proporção do tempo de corrida sob SC ou VSC", "0–1"),
    ("piloto_corrida", "GPsAnteriores", "GPs em que o piloto largou antes desta corrida, carreira inteira (experiência; fonte: API Jolpica/Ergast)", "GPs"),
    ("piloto_corrida", "Idade", "Idade do piloto na data da corrida", "anos"),
    ("piloto_corrida", "MolhadasAnteriores", "Corridas molhadas disputadas antes desta, desde 2021", "corridas"),
]


# ---------- 2.7b Experiência e idade dos pilotos (H3) ----------

# Status da API que NÃO contam como GP disputado (o piloto não largou)
STATUS_NAO_LARGOU = ("did not start", "withdrew", "did not qualify",
                     "did not prequalify", "not qualified")


def largou_na_corrida(c: pd.DataFrame) -> pd.Series:
    """Regra de "largou" (conta como GP disputado) na base da Ergast:
    NÃO largou quando completou 0 voltas E a Ergast marca a ausência de
    largada — posição "W" (retirado antes da largada) ou "F" (não se
    classificou), largada não registrada (grid = 0) ou status de não
    largada / não classificação. Com 1 ou mais voltas, largou (ex.: Norris,
    México 2019, aparece como "W" mas completou 48 voltas). Com 0 voltas e
    posição "R", largou e abandonou na 1ª volta (ex.: acidente na largada)."""
    def col(nome, padrao):
        return c[nome] if nome in c.columns else pd.Series(padrao, index=c.index)
    voltas = pd.to_numeric(col("laps", np.nan), errors="coerce")
    grid = pd.to_numeric(col("grid", np.nan), errors="coerce")
    pos = col("positionText", "").astype(str).str.strip().str.upper()
    st = col("status", "").astype(str).str.lower()
    status_nao = st.apply(lambda x: any(k in x for k in STATUS_NAO_LARGOU))
    sem_volta = voltas.fillna(0) == 0
    nao_largou = sem_volta & (pos.isin(["W", "F"]) | (grid == 0) | status_nao)
    # sem informação de voltas: usa só o status
    nao_largou |= voltas.isna() & status_nao
    return ~nao_largou


def carregar_historico():
    """Lê os arquivos brutos de carreira (01_coleta.py --historico) e devolve:
    {driverId: lista ordenada de chaves temporada*100+etapa dos GPs em que o
    piloto largou} e {driverId: data de nascimento}."""
    largadas, nascimento = {}, {}
    if not os.path.isdir(HISTORICO_DIR):
        return largadas, nascimento
    for arq in os.listdir(HISTORICO_DIR):
        caminho = os.path.join(HISTORICO_DIR, arq)
        if arq.startswith("corridas_") and arq.endswith(".csv"):
            driver_id = arq[len("corridas_"):-4]
            c = pd.read_csv(caminho, low_memory=False)
            largou = largou_na_corrida(c)
            chave = (pd.to_numeric(c["season"], errors="coerce") * 100
                     + pd.to_numeric(c["round"], errors="coerce"))
            largadas[driver_id] = np.sort(chave[largou].dropna().to_numpy())
        elif arq.startswith("info_") and arq.endswith(".csv"):
            driver_id = arq[len("info_"):-4]
            i = pd.read_csv(caminho)
            if "dateOfBirth" in i.columns and len(i):
                nascimento[driver_id] = pd.to_datetime(i["dateOfBirth"].iloc[0], errors="coerce")
    return largadas, nascimento


def completar_driver_id(pc: pd.DataFrame) -> pd.Series:
    """Em algumas corridas a FastF1 entrega os resultados sem o DriverId
    (identificador da API Ergast), normalmente quando a API não respondia na
    hora da coleta. O piloto é o mesmo das outras corridas, então o
    identificador é completado pela sigla (Driver), primeiro dentro da mesma
    temporada e depois em qualquer temporada, desde que a sigla corresponda
    a um único identificador."""
    if "DriverId" in pc.columns:
        ids = pc["DriverId"].astype("string").str.strip()
    else:
        ids = pd.Series(pd.NA, index=pc.index, dtype="string")
    ids = ids.mask(ids.isin(["", "nan", "None", "<NA>"]))
    faltando = ids.isna()
    if faltando.any():
        conhecidos = pd.DataFrame({"Driver": pc["Driver"], "Season": pc["Season"], "Id": ids}).dropna()
        por_temporada = (conhecidos.groupby(["Driver", "Season"])["Id"].agg(lambda x: x.mode().iloc[0]))
        unicos = conhecidos.groupby("Driver")["Id"].nunique()
        geral = (conhecidos[conhecidos["Driver"].isin(unicos[unicos == 1].index)]
                 .groupby("Driver")["Id"].first())
        chave = list(zip(pc.loc[faltando, "Driver"], pc.loc[faltando, "Season"]))
        preenchido = pd.Series([por_temporada.get(k, pd.NA) for k in chave], index=pc.index[faltando],
                               dtype="string")
        preenchido = preenchido.fillna(pc.loc[faltando, "Driver"].map(geral).astype("string"))
        ids.loc[faltando] = preenchido
        corridas_sem = sorted(pc.loc[faltando, "RaceId"].astype(str).unique())
        log(f"DriverId ausente em {int(faltando.sum())} observações ({len(corridas_sem)} corridas: "
            f"{', '.join(corridas_sem[:8])}{'...' if len(corridas_sem) > 8 else ''}); "
            f"{int(preenchido.notna().sum())} completadas pela sigla do piloto.")
    return ids.fillna("").astype(str)


def adicionar_experiencia(pc: pd.DataFrame, corridas: pd.DataFrame) -> pd.DataFrame:
    """Cria, para cada piloto em cada corrida:
      - GPsAnteriores: GPs em que o piloto largou ANTES desta corrida, na
        carreira inteira (inclusive antes de 2021) — medida de experiência;
      - Idade: anos completos (fracionários) na data da corrida;
      - MolhadasAnteriores: corridas molhadas disputadas antes desta, só a
        partir de 2021 (o período dos dados de clima)."""
    largadas, nascimento = carregar_historico()
    pc = pc.copy()
    chave = pc["Season"] * 100 + pc["Round"]
    if not largadas:
        log("AVISO: histórico dos pilotos não encontrado (rode 01_coleta.py --historico). "
            "As variáveis de experiência ficarão vazias.")
        pc["GPsAnteriores"] = np.nan
        pc["Idade"] = np.nan
    else:
        ids = completar_driver_id(pc)
        pc["DriverId"] = ids.replace("", np.nan)
        pc["GPsAnteriores"] = [
            float(np.searchsorted(largadas[d], k, side="left")) if d in largadas else np.nan
            for d, k in zip(ids, chave)]
        datas = pc["RaceId"].map(corridas.set_index("RaceId")["EventDate"])
        datas = pd.to_datetime(datas, errors="coerce")
        nasc = ids.map(nascimento)
        pc["Idade"] = (datas - pd.to_datetime(nasc, errors="coerce")).dt.days / 365.25
        sem = pc["GPsAnteriores"].isna()
        if sem.any():
            log(f"AVISO: {int(sem.sum())} observações sem histórico do piloto "
                f"({', '.join(sorted(pc.loc[sem, 'Driver'].astype(str).unique())[:10])}).")
    # corridas molhadas anteriores (desde 2021), na ordem do calendário
    ordem = pc.assign(_k=chave).sort_values(["Driver", "_k"])
    molhada = ordem["WetRace"].astype(bool).astype(int)
    pc["MolhadasAnteriores"] = (molhada.groupby(ordem["Driver"]).cumsum() - molhada) \
        .reindex(pc.index)
    return pc


# ---------- 2.8 Execução do tratamento ----------

def etapa_tratamento():
    log("===== ETAPA 2 — TRATAMENTO E CRIAÇÃO DAS VARIÁVEIS =====")
    pastas = listar_corridas_brutas()
    if not pastas:
        log("Nenhuma corrida bruta encontrada. Rode primeiro a coleta.")
        return
    os.makedirs(DIAGNOSTICO_DIR, exist_ok=True)

    # 2.8.1 Leitura e tratamento corrida a corrida
    insumos, todas_voltas, todos_eventos = [], [], []
    for pasta in pastas:
        with open(os.path.join(pasta, ARQ_META), encoding="utf-8") as f:
            meta = json.load(f)
        try:
            voltas = ler_bruto(pasta, ARQ_VOLTAS, dtype={"TrackStatus": str,
                                                          "DriverNumber": str})
            if voltas.empty:
                continue
            clima = ler_bruto(pasta, ARQ_CLIMA)
            resultados = ler_bruto(pasta, ARQ_RESULTADOS)
            mensagens = ler_bruto(pasta, ARQ_MENSAGENS)
            status = ler_bruto(pasta, ARQ_STATUS, dtype={"Status": str})
            v = tratar_voltas(voltas, clima, meta)
            ev = tratar_mensagens(mensagens, meta)
            todas_voltas.append(v)
            todos_eventos.append(ev)
            insumos.append({"meta": meta, "resultados": resultados, "eventos": ev,
                            "clima": clima, "status": status})
        except Exception as e:
            log(f"Erro ao tratar {pasta}: {type(e).__name__}: {e}")
    log(f"{len(insumos)} corridas lidas.")

    voltas = pd.concat(todas_voltas, ignore_index=True)
    eventos = pd.concat(todos_eventos, ignore_index=True)

    # 2.8.2 Referência por circuito: mediana das voltas de ritmo em pista
    # seca no mesmo local em todas as temporadas. Permite comparar corridas
    # quentes e frias no mesmo circuito (H1) sem eliminar a variação entre
    # corridas (o que aconteceria comparando só com a própria corrida).
    seco = voltas["IsPaceLap"] & ~voltas["IsWetCompound"] & ~voltas["Rainfall"].fillna(False)
    ref = voltas[seco].groupby("Location")["LapTimeSeconds"].median().rename("CircuitMedianDryLap")
    voltas = voltas.merge(ref, left_on="Location", right_index=True, how="left")
    voltas["LapTimePct_vsCircuit"] = 100 * (voltas["LapTimeSeconds"] / voltas["CircuitMedianDryLap"] - 1)
    voltas["LogLapTime"] = np.log(voltas["LapTimeSeconds"])

    # 2.8.3 Resíduo dentro do stint (consistência)
    voltas["StintResidual"] = np.nan
    ritmo = voltas[voltas["IsPaceLap"]]
    for _, g in ritmo.groupby(["RaceId", "Driver", "Stint"]):
        voltas.loc[g.index, "StintResidual"] = residuo_no_stint(g)

    # 2.8.4 Bases piloto-corrida e corrida
    lista_pc, lista_corridas = [], []
    for ins in insumos:
        meta = ins["meta"]
        v = voltas[voltas["RaceId"] == id_corrida(meta)]
        lista_pc.append(montar_piloto_corrida(v, ins["resultados"], ins["eventos"],
                                              ins["status"], meta))
        lista_corridas.append(montar_corrida(v, ins["clima"], ins["status"],
                                             ins["eventos"], meta))
    piloto_corrida = pd.concat(lista_pc, ignore_index=True)
    corridas = pd.DataFrame(lista_corridas)

    # 2.8.5 Abandono por acidente: escolha e aplicação da aproximação
    tab_proxy, regra_auto = avaliar_regras_proxy(piloto_corrida)
    tab_proxy.to_csv(os.path.join(DIAGNOSTICO_DIR, "validacao_proxy_acidente.csv"), index=False)
    piloto_corrida = piloto_corrida.merge(corridas[["RaceId", "WetRace"]], on="RaceId", how="left")
    validar_proxy_por_condicao(piloto_corrida).round(3).to_csv(
        os.path.join(DIAGNOSTICO_DIR, "validacao_proxy_por_condicao.csv"), index=False)
    piloto_corrida = piloto_corrida.drop(columns=["WetRace"])
    regra = regra_auto if REGRA_PROXY_ACIDENTE == "auto" else REGRA_PROXY_ACIDENTE
    if not tab_proxy.empty:
        sel = tab_proxy.set_index("Regra").loc[regra]
        log(f"Aproximação de acidente usada: {regra} "
            f"({'escolha automática pelo maior F1' if REGRA_PROXY_ACIDENTE == 'auto' else 'definida em REGRA_PROXY_ACIDENTE'}; "
            f"precisão {sel['Precisao']:.0%}, cobertura {sel['Cobertura_recall']:.0%}, "
            f"F1 {sel['F1']:.2f}; base: {sel['N_abandonos_com_causa']} abandonos "
            f"com causa oficial em {sel['Temporadas']}).")
    piloto_corrida["DNF_Accident_proxy"] = REGRAS_PROXY[regra](piloto_corrida).astype(bool) \
        & piloto_corrida["DNF"]
    oficial = piloto_corrida["DNF_Category"].isin(["Accident_or_error", "Mechanical_or_other"])
    piloto_corrida["DNF_Accident"] = np.where(
        oficial, piloto_corrida["DNF_Category"].eq("Accident_or_error"),
        piloto_corrida["DNF_Category"].eq("Retired_generic") & piloto_corrida["DNF_Accident_proxy"])
    piloto_corrida["DNF_Accident"] = piloto_corrida["DNF_Accident"].astype(bool)
    piloto_corrida["DNF_Accident_Source"] = np.select(
        [oficial, piloto_corrida["DNF_Category"].eq("Retired_generic")],
        ["oficial", f"aproximacao_{regra}"], default="nao_se_aplica")

    # 2.8.6 Medidas de erro do piloto (variáveis dependentes de H2).
    # A medida PRINCIPAL não usa o abandono por acidente, porque a causa do
    # abandono só é informada oficialmente em 2021–2022.
    pc = piloto_corrida
    pc["ErrorCount"] = (pc["NumDeletedLaps"] + pc["NumPenalties_driving"]
                        + pc["NumSpinsMsg"] + pc["NumSlowLaps"])
    pc["ErrorRate_per100laps"] = 100 * pc["ErrorCount"] / pc["NumLaps"].replace(0, np.nan)
    pc["AnyDrivingPenalty"] = pc["NumPenalties_driving"] > 0
    pc["AnySeriousError"] = pc["DNF_Accident"] | pc["AnyDrivingPenalty"]

    # Variáveis de corrida repetidas na base piloto-corrida (para os modelos)
    cols_corrida = ["RaceId", "AirTemp_mean", "AirTemp_max", "TrackTemp_mean",
                    "TrackTemp_max", "Humidity_mean", "WindSpeed_mean",
                    "Rainfall_any", "Rainfall_share_of_race", "WetRace",
                    "ShareWetCompoundLaps", "PctTime_SafetyCarOrVSC", "PctTime_RedFlag"]
    pc = pc.merge(corridas[cols_corrida].rename(columns={"ShareWetCompoundLaps": "ShareWetCompoundLaps_race"}),
                  on="RaceId", how="left")

    # Contagens por corrida que dependem da base piloto-corrida
    por_corrida = pc.groupby("RaceId").agg(
        NumDNF_total=("DNF", "sum"), NumDNF_accident=("DNF_Accident", "sum"),
        NumDeletedLaps=("NumDeletedLaps", "sum"), NumPenalties=("NumPenalties", "sum"),
        NumPenalties_driving=("NumPenalties_driving", "sum"))
    corridas = corridas.merge(por_corrida, left_on="RaceId", right_index=True, how="left")

    # 2.8.6b Experiência e idade dos pilotos (H3)
    pc = adicionar_experiencia(pc, corridas)

    # 2.8.7 Identificadores comuns: era, equipe-temporada e amostra principal
    for df in (voltas, pc, corridas):
        df["Era"] = df["Season"].apply(era_regulamento)
        df["MotivoExclusao"] = df["RaceId"].map(CORRIDAS_EXCLUIDAS).fillna("")
        df["AmostraPrincipal"] = df["MotivoExclusao"].eq("")
    for df in (voltas, pc):
        df["TeamSeason"] = df["Team"].astype(str) + "_" + df["Season"].astype(str)

    # ---------- 2.8.8 Salvando as bases tratadas ----------
    col_voltas = [
        "RaceId", "Season", "Round", "EventName", "Location", "Era",
        "AmostraPrincipal", "Driver", "DriverNumber", "Team", "TeamSeason",
        "LapNumber", "Stint", "Position", "LapTimeSeconds", "LogLapTime",
        "LapTimePct_vsRace", "LapTimePct_vsCircuit", "LapTimePct_vsWindow",
        "StintResidual",
        "Sector1Time_s", "Sector2Time_s", "Sector3Time_s", "Compound",
        "TyreLife", "FreshTyre", "IsWetCompound", "TrackStatus",
        "IsNeutralized", "IsYellow", "IsPitInLap", "IsPitOutLap", "IsPitLap",
        "IsLap1", "IsCleanLap", "IsSlowLap", "IsPaceLap", "Deleted",
        "DeletedReason", "IsAccurate", "AirTemp", "TrackTemp", "Humidity",
        "Pressure", "Rainfall", "WindSpeed", "WindDirection",
        "LapStartTime_s", "Time_s",
    ]
    voltas = voltas[[c for c in col_voltas if c in voltas.columns]]
    voltas = voltas.sort_values(["Season", "Round", "Driver", "LapNumber"])
    voltas.to_csv(os.path.join(TRATADOS_DIR, "voltas_tratadas.csv"), index=False)
    pc.sort_values(["Season", "Round", "Driver"]).to_csv(
        os.path.join(TRATADOS_DIR, "piloto_corrida.csv"), index=False)
    corridas.sort_values(["Season", "Round"]).to_csv(
        os.path.join(TRATADOS_DIR, "corridas.csv"), index=False)
    eventos.to_csv(os.path.join(TRATADOS_DIR, "eventos_direcao_prova.csv"), index=False)
    pd.DataFrame(DICIONARIO, columns=["Base", "Variavel", "Descricao", "Unidade"]).to_csv(
        os.path.join(TRATADOS_DIR, "dicionario_variaveis.csv"), index=False)

    # ---------- 2.8.9 Diagnósticos (conferir antes de analisar) ----------
    # 1) Valores de Status por temporada e como foram classificados
    (pc.groupby(["Season", "Status", "DNF_Category"], dropna=False).size()
     .rename("N").reset_index()
     .to_csv(os.path.join(DIAGNOSTICO_DIR, "status_resultados_por_temporada.csv"), index=False))
    # 2) Motivos dos eventos e sua classificação (revisar a classe "outros")
    if not eventos.empty:
        (eventos.groupby(["TipoEvento", "ClasseMotivo", "Motivo"], dropna=False).size()
         .rename("N").reset_index().sort_values("N", ascending=False)
         .to_csv(os.path.join(DIAGNOSTICO_DIR, "motivos_eventos.csv"), index=False))
        (eventos.groupby(["TipoEvento", "ClasseMotivo"]).size().rename("N").reset_index()
         .to_csv(os.path.join(DIAGNOSTICO_DIR, "eventos_por_tipo_e_classe.csv"), index=False))
    # 3) Origem da informação de abandono por acidente, por temporada
    (pc[pc["DNF"]].groupby(["Season", "DNF_Accident_Source", "DNF_Accident"]).size()
     .rename("N").reset_index()
     .to_csv(os.path.join(DIAGNOSTICO_DIR, "abandonos_por_origem.csv"), index=False))
    # 4) Experiência por piloto: conferir com uma fonte pública (ex.: nº de
    #    GPs do piloto antes da 1ª corrida da amostra)
    if pc["GPsAnteriores"].notna().any():
        (pc.sort_values(["Season", "Round"]).groupby("Driver")
         .agg(DriverId=("DriverId", "first"), Primeira_corrida=("RaceId", "first"),
              GPs_antes_da_primeira=("GPsAnteriores", "first"),
              Ultima_corrida=("RaceId", "last"), GPs_antes_da_ultima=("GPsAnteriores", "last"),
              Idade_na_primeira=("Idade", "first"))
         .round(1).reset_index().sort_values("GPs_antes_da_primeira", ascending=False)
         .to_csv(os.path.join(DIAGNOSTICO_DIR, "experiencia_pilotos.csv"), index=False))
    # 5) Resumo da amostra
    amostra = corridas["AmostraPrincipal"]
    resumo = pd.DataFrame([{
        "Corridas": corridas["RaceId"].nunique(),
        "Corridas_amostra_principal": int(amostra.sum()),
        "Temporadas": ", ".join(map(str, sorted(corridas["Season"].unique()))),
        "Voltas_total": len(voltas),
        "Voltas_ritmo": int(voltas["IsPaceLap"].sum()),
        "Observacoes_piloto_corrida": len(pc),
        "Corridas_molhadas": int(corridas["WetRace"].sum()),
        "Abandonos": int(pc["DNF"].sum()),
        "Abandonos_acidente": int(pc["DNF_Accident"].sum()),
        "Regra_proxy_acidente": regra,
        "Voltas_sem_clima": int(voltas["AirTemp"].isna().sum()),
        "Obs_com_experiencia": int(pc["GPsAnteriores"].notna().sum()),
    }])
    resumo.to_csv(os.path.join(DIAGNOSTICO_DIR, "resumo_amostra.csv"), index=False)

    log(f"Tratamento concluído: {resumo.iloc[0].to_dict()}")
    log(f"Bases salvas em {TRATADOS_DIR}")


# =====================================================================
# EXECUÇÃO
# =====================================================================

if __name__ == "__main__":
    os.makedirs(TRATADOS_DIR, exist_ok=True)
    etapa_tratamento()
    log("Próximo passo: python src/03_exploracao.py")