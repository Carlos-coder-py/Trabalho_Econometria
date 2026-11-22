"""
O QUE FAZ
    Obtém cada Grande Prêmio de 2021 a 2026 pela biblioteca FastF1 e salva
    as tabelas EXATAMENTE como a biblioteca as entrega, sem nenhum
    tratamento. É a "versão original" da base exigida na entrega.

    Para cada corrida é criada a pasta dados/brutos/corridas/<ano>/<etapa>_<gp>/
    com: voltas.csv, clima.csv, resultados.csv, mensagens_direcao_prova.csv,
    status_pista.csv e meta.json (identificação do evento).

    - Pode ser executado várias vezes: corridas já salvas são puladas.
    - Corridas sem resultado oficial (API recusou) não são salvas e são
      tentadas de novo na execução seguinte.
    - Se o limite da API for atingido, para e mantém o que já foi salvo.

    HISTÓRICO DOS PILOTOS (opção --historico, usado na H3)
    Para cada piloto que aparece nas corridas coletadas, baixa da API
    Jolpica/Ergast (a mesma fonte dos resultados oficiais) a lista de TODAS
    as corridas da carreira e a data de nascimento. Salva sem tratamento em
    dados/brutos/historico_pilotos/:
        corridas_<driverId>.csv -> uma linha por GP disputado na carreira
        info_<driverId>.csv     -> dados do piloto (inclui data de nascimento)
    O tratamento usa esses arquivos para calcular a experiência do piloto
    (nº de GPs disputados antes de cada corrida) e a idade.

COMO RODAR (terminal do VS Code, na pasta raiz do projeto)
    python src/01_coleta.py                    -> coleta tudo o que faltar
    python src/01_coleta.py --offline          -> usa somente o cache local
    python src/01_coleta.py --teste            -> coleta só uma corrida
    python src/01_coleta.py --anos 2023 2024   -> só as temporadas indicadas
    python src/01_coleta.py --historico        -> só o histórico dos pilotos

PRÓXIMO PASSO
    python src/02_tratamento.py
"""

import argparse
import json
import logging
import os
import re
import shutil
import time
import unicodedata
import warnings
from datetime import datetime

import pandas as pd
import fastf1


# =====================================================================
# 0. CONFIGURAÇÃO DA COLETA
# =====================================================================

# Pasta raiz do projeto: a pasta acima de src/ (ou a própria pasta, se os
# códigos estiverem soltos na raiz). É onde ficam f1_cache/, dados/ e
# resultados/.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
DADOS_DIR = os.path.join(BASE_DIR, "dados")

# Cache da FastF1. Aceita "f1_cache" ou "f1-cache" (usa a que já existir).
CACHE_DIR = next(
    (os.path.join(BASE_DIR, n) for n in ("f1_cache", "f1-cache")
     if os.path.isdir(os.path.join(BASE_DIR, n))),
    os.path.join(BASE_DIR, "f1_cache"),
)
BRUTOS_DIR = os.path.join(DADOS_DIR, "brutos")
BRUTOS_CORRIDAS_DIR = os.path.join(BRUTOS_DIR, "corridas")
BRUTOS_CALENDARIO_DIR = os.path.join(BRUTOS_DIR, "calendario")
FALHAS_CSV = os.path.join(BRUTOS_DIR, "falhas_coleta.csv")

# Nomes dos arquivos brutos salvos para cada corrida
ARQ_VOLTAS, ARQ_CLIMA = "voltas.csv", "clima.csv"
ARQ_RESULTADOS, ARQ_MENSAGENS = "resultados.csv", "mensagens_direcao_prova.csv"
ARQ_STATUS, ARQ_META = "status_pista.csv", "meta.json"

# Escopo da coleta
SEASONS = list(range(2021, 2027))   # 2021 a 2026 (o range exclui o 2027)
SESSION_TYPE = "R"                  # apenas corridas principais (sem Sprint)
CORRIDA_TESTE = (2024, 1)           # corrida usada com a opção --teste
PAUSA_ENTRE_CORRIDAS_S = 1.0        # respeita o limite da API (500 chamadas/hora)


# Funções auxiliares
def log(msg: str):
    """Imprime uma mensagem com horário (acompanha o andamento)."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def slug(texto: str) -> str:
    """Converte o nome do GP em nome de pasta (sem acentos ou espaços)."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()


def pasta_corrida(ano: int, etapa: int, nome_evento: str) -> str:
    """Pasta dos dados brutos de uma corrida: dados/brutos/corridas/<ano>/<etapa>_<gp>/"""
    return os.path.join(BRUTOS_CORRIDAS_DIR, str(ano),
                        f"{int(etapa):02d}_{slug(nome_evento)}")


def corrida_ja_coletada(pasta: str) -> bool:
    """A corrida só conta como coletada se o meta.json existir (ele é o
    último arquivo escrito, então uma coleta interrompida não é aceita)."""
    return os.path.exists(os.path.join(pasta, ARQ_META))

warnings.filterwarnings("ignore")


# =====================================================================
# 0.1 CONFIGURAÇÃO DA FASTF1 (logs, exceções e cache)
# =====================================================================

def silenciar_logs_fastf1():
    """A FastF1 imprime muitas mensagens técnicas (INFO, avisos e tracebacks
    de tentativas que ela mesma contorna). Elas poluem o terminal e escondem
    o que importa, então só erros graves são mostrados."""
    try:
        fastf1.set_log_level("ERROR")
    except Exception:
        pass
    for nome in list(logging.root.manager.loggerDict):
        if nome.startswith(("fastf1", "requests_cache", "urllib3")):
            lg = logging.getLogger(nome)
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False


silenciar_logs_fastf1()


class ResultadosIncompletosError(Exception):
    """O resultado oficial (posição, largada, status) vem da API Jolpica/
    Ergast. Quando ela recusa a requisição ('Too Many Requests'), a FastF1
    segue sem esses campos. Nesse caso a corrida NÃO é salva, para não
    gravar uma base bruta incompleta; ela é tentada de novo na próxima
    execução."""

# Exceções da FastF1 (os nomes mudaram entre versões; por isso o try/except)
try:
    from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError
except ImportError:  # versões antigas
    from fastf1.core import DataNotLoadedError
    try:
        from fastf1.req import RateLimitExceededError
    except ImportError:
        class RateLimitExceededError(Exception):
            pass


def configurar_cache(offline: bool = False):
    """Ativa o cache local da FastF1 (evita baixar a mesma corrida 2 vezes)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if hasattr(fastf1.Cache, "configure"):          # FastF1 >= 3.8
        fastf1.Cache.configure(cache_dir=CACHE_DIR)
    else:                                           # versões anteriores
        fastf1.Cache.enable_cache(CACHE_DIR)
    if offline:
        fastf1.Cache.offline_mode(True)
    silenciar_logs_fastf1()   # de novo: alguns registradores surgem só agora
    log(f"Cache da FastF1: {CACHE_DIR}" + (" (modo offline)" if offline else ""))


# =====================================================================
# 1. COLETA CORRIDA A CORRIDA
# =====================================================================

def registrar_falha(ano, etapa, evento, erro):
    os.makedirs(BRUTOS_DIR, exist_ok=True)
    linha = pd.DataFrame([{
        "DataHora": datetime.now().isoformat(timespec="seconds"),
        "Season": ano, "Round": etapa, "EventName": evento, "Erro": erro,
    }])
    linha.to_csv(FALHAS_CSV, mode="a", index=False,
                 header=not os.path.exists(FALHAS_CSV))


def obter_calendario(ano: int) -> pd.DataFrame:
    """Baixa o calendário da temporada e salva a versão bruta."""
    cal = fastf1.get_event_schedule(ano, include_testing=False)
    os.makedirs(BRUTOS_CALENDARIO_DIR, exist_ok=True)
    pd.DataFrame(cal).to_csv(
        os.path.join(BRUTOS_CALENDARIO_DIR, f"calendario_{ano}.csv"), index=False)
    return cal


def corrida_ja_aconteceu(linha_cal) -> bool:
    """Evita tentar baixar corridas de 2026 que ainda não ocorreram.
    Considera a corrida disponível 6 horas após a largada (horário UTC)."""
    data = linha_cal.get("Session5DateUtc", pd.NaT)
    if pd.isna(data):
        data = linha_cal.get("EventDate", pd.NaT)
    if pd.isna(data):
        return True
    data = pd.Timestamp(data)
    if data.tzinfo is not None:
        data = data.tz_convert(None)
    agora_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return data + pd.Timedelta(hours=6) < agora_utc


def _tabela_ou_vazia(sessao, atributo: str) -> pd.DataFrame:
    """Devolve a tabela da sessão; se a FastF1 não a carregou, devolve
    uma tabela vazia (a ausência fica registrada no meta.json)."""
    try:
        tab = getattr(sessao, atributo)
        return pd.DataFrame(tab) if tab is not None else pd.DataFrame()
    except (DataNotLoadedError, AttributeError, KeyError):
        return pd.DataFrame()


def coletar_corrida(ano: int, etapa: int, linha_cal) -> str:
    """Carrega uma corrida pela FastF1 e salva suas tabelas brutas."""
    nome_evento = linha_cal["EventName"]
    pasta_final = pasta_corrida(ano, etapa, nome_evento)
    pasta_tmp = pasta_final + "_incompleta"

    sessao = fastf1.get_session(ano, int(etapa), SESSION_TYPE)
    # telemetry=False: a telemetria de 6 temporadas é muito pesada e não
    # entra nos modelos; voltas, clima, resultados e mensagens bastam.
    sessao.load(laps=True, weather=True, telemetry=False, messages=True)

    tabelas = {
        ARQ_VOLTAS: _tabela_ou_vazia(sessao, "laps"),
        ARQ_CLIMA: _tabela_ou_vazia(sessao, "weather_data"),
        ARQ_RESULTADOS: _tabela_ou_vazia(sessao, "results"),
        ARQ_MENSAGENS: _tabela_ou_vazia(sessao, "race_control_messages"),
        ARQ_STATUS: _tabela_ou_vazia(sessao, "track_status"),
    }
    if tabelas[ARQ_VOLTAS].empty:
        raise DataNotLoadedError("Sem dados de voltas para esta corrida.")
    res = tabelas[ARQ_RESULTADOS]
    if res.empty or "Status" not in res.columns or res["Status"].isna().all():
        raise ResultadosIncompletosError(
            "Resultado oficial indisponível (API Jolpica/Ergast recusou ou "
            "não respondeu). Corrida não salva; será tentada de novo.")

    # Escreve tudo numa pasta temporária e só depois renomeia: se o script
    # for interrompido no meio, não sobra uma corrida pela metade.
    if os.path.exists(pasta_tmp):
        shutil.rmtree(pasta_tmp)
    os.makedirs(pasta_tmp)
    for nome_arq, tab in tabelas.items():
        tab.to_csv(os.path.join(pasta_tmp, nome_arq), index=False)

    def _txt(v):
        try:
            return None if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
        except (TypeError, ValueError):
            return str(v)

    def _attr(nome):
        """Lê uma informação da sessão sem quebrar a coleta se ela não
        existir (ex.: t0_date, session_start_time, total_laps). Na v3.0
        essa leitura gerava o erro 'DataNotLoadedError' em todas as corridas."""
        try:
            return _txt(getattr(sessao, nome))
        except Exception:
            return None

    meta = {
        "Season": ano,
        "Round": int(etapa),
        "EventName": nome_evento,
        "OfficialEventName": _txt(linha_cal.get("OfficialEventName")),
        "Country": _txt(linha_cal.get("Country")),
        "Location": _txt(linha_cal.get("Location")),
        "EventDate": _txt(linha_cal.get("EventDate")),
        "EventFormat": _txt(linha_cal.get("EventFormat")),
        "t0_date": _attr("t0_date"),
        "session_start_time": _attr("session_start_time"),
        "total_laps": _attr("total_laps"),
        "tabelas_vazias": [k for k, v in tabelas.items() if v.empty],
        "versao_fastf1": fastf1.__version__,
        "coletado_em": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(pasta_tmp, ARQ_META), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if os.path.exists(pasta_final):
        shutil.rmtree(pasta_final)
    os.replace(pasta_tmp, pasta_final)
    return pasta_final


def etapa_coleta(anos=None, teste=False, offline=False):
    """Percorre as temporadas e coleta todas as corridas ainda não salvas.
    Pode ser executada várias vezes: retoma de onde parou."""
    log("===== ETAPA 1 — COLETA DOS DADOS BRUTOS =====")
    anos = [CORRIDA_TESTE[0]] if teste else (anos or SEASONS)
    n_novas = n_puladas = n_falhas = n_sem_resultado = 0
    parar = False

    for ano in anos:
        if parar:
            break
        try:
            calendario = obter_calendario(ano)
        except RateLimitExceededError:
            log("Limite da API atingido ao baixar o calendário. Rode de novo "
                "em ~1 hora; o que já foi salvo será mantido.")
            return
        except Exception as e:
            log(f"Calendário de {ano} indisponível ({e}). Pulando o ano.")
            continue

        for _, linha in calendario.iterrows():
            etapa = int(linha["RoundNumber"])
            if etapa == 0:
                continue
            if teste and etapa != CORRIDA_TESTE[1]:
                continue
            evento = linha["EventName"]
            pasta = pasta_corrida(ano, etapa, evento)

            if corrida_ja_coletada(pasta):
                n_puladas += 1
                continue
            if not corrida_ja_aconteceu(linha):
                log(f"{ano} R{etapa:02d} {evento}: ainda não aconteceu, pulando.")
                continue

            log(f"Coletando {ano} R{etapa:02d} — {evento} ...")
            try:
                coletar_corrida(ano, etapa, linha)
                n_novas += 1
                n_sem_resultado = 0
                log("   -> OK, dados brutos salvos.")
            except RateLimitExceededError:
                log("Limite da API atingido (500 chamadas/hora). Tudo que já "
                    "foi coletado está salvo. Rode o script de novo em ~1 hora: "
                    "ele continua de onde parou.")
                return
            except ResultadosIncompletosError as e:
                n_sem_resultado += 1
                log("   -> sem resultado oficial; tentará de novo na próxima execução.")
                registrar_falha(ano, etapa, evento, str(e))
                n_falhas += 1
                # Três seguidas = a cota da API de resultados acabou. Parar
                # evita gastar a cota da próxima hora com tentativas inúteis.
                if not offline and n_sem_resultado >= 3:
                    log("A API de resultados (Jolpica/Ergast) está recusando "
                        "pedidos ('Too Many Requests'). Coleta interrompida "
                        "para não gastar a cota. Aguarde ~1 hora e rode de novo.")
                    parar = True
                    break
                time.sleep(PAUSA_ENTRE_CORRIDAS_S)
                continue
            except Exception as e:
                log(f"   -> falhou ({type(e).__name__}: {e}).")
                registrar_falha(ano, etapa, evento, f"{type(e).__name__}: {e}")
                n_falhas += 1
            time.sleep(PAUSA_ENTRE_CORRIDAS_S)

    log(f"Coleta concluída: {n_novas} novas, {n_puladas} já existentes, "
        f"{n_falhas} falhas (ver {FALHAS_CSV}).")


# =====================================================================
# 2. HISTÓRICO DE CARREIRA DOS PILOTOS (H3 — experiência)
# =====================================================================

HISTORICO_DIR = os.path.join(BRUTOS_DIR, "historico_pilotos")


def pilotos_da_amostra() -> list:
    """Lista os identificadores (driverId da API Ergast) dos pilotos que
    aparecem nos resultados brutos já coletados."""
    ids = set()
    for raiz, _, arquivos in os.walk(BRUTOS_CORRIDAS_DIR):
        if ARQ_RESULTADOS in arquivos:
            try:
                r = pd.read_csv(os.path.join(raiz, ARQ_RESULTADOS), usecols=["DriverId"])
                ids.update(r["DriverId"].dropna().astype(str))
            except (ValueError, pd.errors.EmptyDataError):
                continue
    return sorted(i for i in ids if i and i.lower() != "nan")


def baixar_carreira(ergast, driver_id: str) -> pd.DataFrame:
    """Todas as corridas da carreira de um piloto (uma linha por GP), página
    por página (a API devolve no máximo 100 resultados por consulta)."""
    linhas, offset = [], 0
    while True:
        resp = ergast.get_race_results(driver=driver_id, limit=100, offset=offset)
        if resp.description.empty:
            break
        for i, corrida in resp.description.reset_index(drop=True).iterrows():
            res = resp.content[i]
            if res.empty:
                continue
            linha = corrida.to_dict()
            for c in ("number", "position", "positionText", "points", "grid", "laps",
                      "status", "constructorId", "constructorName"):
                if c in res.columns:
                    linha[c] = res.iloc[0][c]
            linhas.append(linha)
        offset += 100
        if offset >= resp.total_results:
            break
    return pd.DataFrame(linhas)


TENTATIVAS_429 = 4      # tentativas quando a API responde "Too Many Requests"
ESPERA_429_S = 60       # espera (segundos) antes de tentar de novo


def coletar_historico():
    """Baixa o histórico de carreira e a data de nascimento de cada piloto.
    Pilotos já salvos são pulados; se o limite da API for atingido, basta
    rodar de novo depois."""
    from fastf1.ergast import Ergast
    log("===== HISTÓRICO DE CARREIRA DOS PILOTOS (H3) =====")
    os.makedirs(HISTORICO_DIR, exist_ok=True)
    ids = pilotos_da_amostra()
    if not ids:
        log("Nenhum piloto encontrado nos resultados brutos (coluna DriverId). "
            "Rode primeiro a coleta das corridas.")
        return
    log(f"{len(ids)} pilotos na amostra.")
    ergast = Ergast(result_type="pandas", auto_cast=True)
    novos = falhas = 0
    for driver_id in ids:
        arq_corridas = os.path.join(HISTORICO_DIR, f"corridas_{driver_id}.csv")
        arq_info = os.path.join(HISTORICO_DIR, f"info_{driver_id}.csv")
        if os.path.exists(arq_corridas) and os.path.exists(arq_info):
            continue
        try:
            # A API recusa rajadas de pedidos ("Too Many Requests", erro 429):
            # espera e tenta de novo, até TENTATIVAS_429 vezes.
            for tentativa in range(1, TENTATIVAS_429 + 1):
                try:
                    carreira = baixar_carreira(ergast, driver_id)
                    info = pd.DataFrame(ergast.get_driver_info(driver=driver_id))
                    break
                except RateLimitExceededError:
                    raise
                except Exception as e:
                    if "too many requests" not in str(e).lower() or tentativa == TENTATIVAS_429:
                        raise
                    log(f"  {driver_id}: API pediu pausa (Too Many Requests); "
                        f"aguardando {ESPERA_429_S} s (tentativa {tentativa}/{TENTATIVAS_429})...")
                    time.sleep(ESPERA_429_S)
            if carreira.empty or info.empty:
                raise ValueError("resposta vazia da API")
            carreira.to_csv(arq_corridas + ".tmp", index=False)
            info.to_csv(arq_info, index=False)
            os.replace(arq_corridas + ".tmp", arq_corridas)   # grava por último
            novos += 1
            log(f"  {driver_id}: {len(carreira)} GPs na carreira — salvo.")
        except RateLimitExceededError:
            log("Limite da API atingido. O que já foi salvo está mantido; rode de novo "
                "em ~1 hora com --historico.")
            return
        except Exception as e:
            falhas += 1
            log(f"  {driver_id}: falhou ({type(e).__name__}: {e}).")
        time.sleep(PAUSA_ENTRE_CORRIDAS_S)
    log(f"Histórico concluído: {novos} pilotos novos, {falhas} falhas "
        f"(arquivos em {HISTORICO_DIR}).")
    if falhas:
        log("Rode de novo 'python 01_coleta.py --historico': os pilotos já salvos são "
            "pulados e só os que falharam são baixados.")


# =====================================================================
# EXECUÇÃO
# =====================================================================

def main():
    p = argparse.ArgumentParser(description="Etapa 1 — coleta dos dados brutos (FastF1)")
    p.add_argument("--anos", nargs="+", type=int, help="temporadas a coletar")
    p.add_argument("--teste", action="store_true", help="coleta só uma corrida")
    p.add_argument("--offline", action="store_true", help="usa apenas o cache local")
    p.add_argument("--historico", action="store_true",
                   help="baixa só o histórico de carreira dos pilotos (H3)")
    args = p.parse_args()

    os.makedirs(BRUTOS_CORRIDAS_DIR, exist_ok=True)
    configurar_cache(args.offline)
    if args.historico:
        coletar_historico()
    else:
        etapa_coleta(args.anos, args.teste, args.offline)
    log("Próximo passo: python 02_tratamento.py")


if __name__ == "__main__":
    main()