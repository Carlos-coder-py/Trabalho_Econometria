# Trabalho Econometria

# Clima e desempenho na Fórmula 1

**O efeito das variáveis climáticas sobre o desempenho e a taxa de erro dos pilotos de Fórmula 1**

Este repositório reúne o código e a documentação usados no artigo: coleta, tratamento, análise
exploratória e modelos econométricos. Todos os números, tabelas e figuras do artigo podem ser
reproduzidos a partir dele.

---

## Sumário

1. [Pergunta de pesquisa e hipóteses](#1-pergunta-de-pesquisa-e-hipóteses)
2. [Dados](#2-dados)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Como reproduzir](#4-como-reproduzir)
5. [Estratégia empírica](#5-estratégia-empírica)
6. [Principais decisões metodológicas](#6-principais-decisões-metodológicas)
7. [Saídas: tabelas e figuras](#7-saídas-tabelas-e-figuras)
8. [Resultados principais](#8-resultados-principais)
9. [Limitações conhecidas](#9-limitações-conhecidas)
10. [Ambiente e versões](#10-ambiente-e-versões)
11. [Créditos e uso dos dados](#11-créditos-e-uso-dos-dados)

---

## 1. Pergunta de pesquisa e hipóteses

**Pergunta de pesquisa:** qual é o efeito das variações nas condições climáticas, especificamente a
ocorrência de chuva e as oscilações de temperatura, sobre o desempenho e a taxa de erro dos pilotos
de Fórmula 1 durante as corridas?

| Hipótese | Tema | Variável climática | Unidade de análise | Modelo |
|---|---|---|---|---|
| **H1** | Temperatura da pista × ritmo e consistência, com termo quadrático (não linearidade) | Temperatura da pista (°C) | piloto-corrida (corridas secas) e volta | Tabelas M1 e M2 |
| **H2** | Chuva × cada tipo de erro de pilotagem, analisados separadamente | Corrida molhada (0/1) | piloto-corrida | Tabela M3 |
| **H3** | A experiência do piloto modera o efeito da chuva (interação chuva × experiência) | Corrida molhada × log(1 + GPs anteriores) | piloto-corrida | Tabelas M4 e M4b |

O enunciado completo e a fundamentação de cada hipótese estão no artigo.

---

## 2. Dados

| Item | Descrição |
|---|---|
| **Fontes** | Biblioteca [FastF1](https://docs.fastf1.dev) 3.9.0: cronometragem oficial volta a volta, estação meteorológica do circuito, mensagens da direção de prova e status da pista. API [Jolpica/Ergast](https://github.com/jolpica/jolpica-f1): resultados e histórico de carreira dos pilotos |
| **Período** | Temporadas 2021 a 2026. Para 2026, até a etapa 20, última corrida disponível na data da coleta. Apenas corridas principais (sem Sprint) |
| **Tipo** | Dados secundários em painel: piloto × corrida e volta × piloto × corrida |
| **Amostra principal** | 133 corridas (21 molhadas), 38 pilotos, 2.675 observações piloto-corrida e 127.666 voltas de ritmo |
| **Exclusão** | GP da Bélgica 2021: encerrado após poucas voltas atrás do Safety Car, sem disputa em bandeira verde |

### Variáveis principais

| Grupo | Variável | Definição resumida |
|---|---|---|
| Desempenho | `AvgLapTimePct_vsCircuit_w` | Ritmo médio do piloto, em % acima da referência seca do circuito |
| Desempenho | `AvgLapTimePct_vsRace_w` | Ritmo médio em % em relação à mediana do grid na mesma corrida |
| Consistência | `ConsistencyResidStd_w`, `Inconsistencia_janela_w` | Desvio-padrão do ritmo no stint e em relação ao grid nas voltas n±2 |
| Erro (H2) | `PerdaControle` | Rodada (mensagem da direção de prova) ou abandono por acidente (0/1) |
| Erro (H2) | `VoltasLentas_100` | Voltas limpas > 10% mais lentas que a mediana dos outros pilotos nas voltas n±2, no mesmo tipo de pneu, por 100 voltas |
| Erro (H2) | `VoltasAnuladas_100` | Voltas anuladas por limite de pista, por 100 voltas |
| Erro (H2) | `PenalidadePilotagem` | Penalidade por infração de pilotagem (0/1) |
| Clima | `TrackTemp_driver_mean`, `Molhada`, `ShareRainLaps` | Temperatura da pista nas voltas do piloto; corrida molhada; proporção de voltas com chuva |
| Experiência (H3) | `GPsAnteriores`, `Idade`, `MolhadasAnteriores` | GPs em que o piloto largou antes da corrida (carreira inteira); idade na data da corrida; corridas molhadas anteriores desde 2021 |

O dicionário completo é gerado em `dados/tratados/dicionario_variaveis.csv`.

### Disponibilidade dos dados

As pastas `dados/` e `f1_cache/` não são versionadas no Git porque são grandes e podem ser recriadas
pelos códigos. As bases usadas no artigo estão disponíveis em:

- **Dados brutos, tratados e finais:** `[dados.zip na Release v1.0](https://github.com/Carlos-coder-py/Trabalho_Econometria/releases/tag/v1.0)`

Para recriá-las do zero, siga a seção 4.

---

## 3. Estrutura do repositório

```
f1-clima-econometria/
├── src/
│   ├── 01_coleta.py          # Etapa 1: coleta dos dados brutos (FastF1 e Jolpica/Ergast)
│   ├── 02_tratamento.py      # Etapa 2: limpeza, validação e criação das variáveis
│   ├── 03_exploracao.py      # Etapa 3: descritivas, outliers, testes preliminares e figuras
│   └── 04_modelos.py         # Etapa 4: modelos econométricos e testes de H1, H2 e H3
├── dados/                    # gerado pelos códigos (não versionado)
│   ├── brutos/
│   │   ├── corridas/<ano>/<etapa>_<gp>/   # voltas, clima, resultados, mensagens, status, meta.json
│   │   ├── calendario/                    # calendário de cada temporada
│   │   └── historico_pilotos/             # carreira e data de nascimento de cada piloto (H3)
│   ├── tratados/             # bases tratadas (voltas, piloto-corrida, corridas, eventos)
│   │   └── diagnostico/      # validações e conferências (proxy de acidente, experiência etc.)
│   └── final/                # bases de análise lidas pelos modelos
├── resultados/               # gerado pelos códigos
│   ├── tabelas/              # tabelas_exploracao.xlsx, tabelas_modelos.xlsx e CSVs
│   ├── figuras/              # F01–F13 (PNG, 300 dpi)
│   ├── resumo_exploracao.txt
│   └── resumo_modelos.txt
├── f1_cache/                 # cache da FastF1 (não versionado)
├── requirements.txt
├── .gitignore
└── README.md
```

**Fluxo:** cada etapa lê apenas o que a etapa anterior gravou em disco.

```
01_coleta ──► dados/brutos ──► 02_tratamento ──► dados/tratados ──► 03_exploracao ──► dados/final ──► 04_modelos
                                                                         │                                 │
                                                                         └────────► resultados/ ◄──────────┘
```

Cada código começa por uma **seção 0** com os caminhos e os parâmetros da etapa, que são as decisões
metodológicas. Isso permite refazer uma etapa isoladamente, por exemplo refazer o tratamento sem
coletar de novo. Os códigos funcionam dentro de `src/` ou soltos na pasta raiz.

---

## 4. Como reproduzir

### 4.1 Ambiente

```bash
git clone <URL-do-repositório>
cd f1-clima-econometria
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Execução, na ordem

```bash
python src/01_coleta.py              # coleta as corridas (retoma de onde parou)
python src/01_coleta.py --historico  # coleta o histórico de carreira dos pilotos (H3)
python src/02_tratamento.py          # gera dados/tratados/
python src/03_exploracao.py          # gera dados/final/ e resultados/ (tabelas T e figuras F01–F09)
python src/04_modelos.py             # gera tabelas M1–M5 e figuras F10–F13
```

### 4.3 Opções da coleta

| Comando | O que faz |
|---|---|
| `python src/01_coleta.py --teste` | Coleta só uma corrida, para testar o ambiente |
| `python src/01_coleta.py --anos 2023 2024` | Coleta só as temporadas indicadas |
| `python src/01_coleta.py --offline` | Usa apenas o cache local, sem internet |
| `python src/01_coleta.py --historico` | Coleta só o histórico de carreira dos pilotos |

### 4.4 Observações

- **Tempo de execução:** a coleta completa leva algumas horas por causa do limite da API (500
  chamadas por hora). Se o limite for atingido, o código para, mantém o que já foi salvo e continua
  de onde parou na execução seguinte. Os demais códigos levam poucos minutos. O `04_modelos.py` é o
  mais demorado, por causa das 199 permutações.
- **Reprodutibilidade:** o teste de permutação usa a semente fixa 2026. A única fonte de variação
  entre execuções é a própria coleta, porque a FastF1 pode corrigir dados oficiais e a temporada 2026
  ainda está em andamento.

---

## 5. Estratégia empírica

**Especificação base (nível piloto-corrida):**

```
Y_ij = β·Clima_ij + γ·Controles_j + α_piloto + α_circuito + α_equipe×temporada + ε_ij
```

| Elemento | Implementação |
|---|---|
| **Estimador** | Mínimos Quadrados Ordinários (MQO). Para variáveis 0/1, modelo de probabilidade linear |
| **Efeitos fixos** | Piloto; circuito (separado em antes e depois quando o traçado mudou: Barcelona e Marina Bay, 2023); equipe × temporada (carro de cada ano, absorvendo as mudanças de regulamento de 2022 e 2026) |
| **Cálculo** | Efeitos fixos absorvidos pela transformação *within* (projeções alternadas, teorema de Frisch–Waugh–Lovell), equivalente à regressão com todas as dummies |
| **Erros-padrão** | Agrupados (cluster) por corrida, com correção de graus de liberdade pelos efeitos fixos. Inferência pela distribuição *t* com (nº de corridas − 1) graus de liberdade |
| **H1** | Temperatura da pista centrada na média e termo quadrático, em corridas secas, com umidade e vento como controles. Robustez no nível da volta, com idade do pneu, número da volta e composto |
| **H2** | Uma regressão para cada tipo de erro, com corrida molhada (0/1) e temperatura da pista |
| **H3** | `Y = δ·Molhada + θ·(Molhada × Exp) + φ·Exp + β·Temp + efeitos fixos`, com Exp = log(1 + GPs anteriores), centrada. H3 prevê θ < 0 (teste unilateral). Reporta o efeito da chuva (δ + θ·Exp) nos percentis 10 e 90 de experiência |
| **Sem controles ruins** | Safety Car, pneu de chuva e posição de largada **não** entram como controles, por serem consequências do clima |

**Testes de robustez:**

| Hipótese | Testes |
|---|---|
| H1 | Temperatura do ar; temperatura média da corrida; ritmo relativo ao grid; sem 2026; sem efeito fixo de piloto |
| H2 | Só 2021–2022, com causa oficial do abandono; proporção de voltas com chuva; sem 2026; Poisson com exposição em voltas; limites de volta lenta de 8% e 15% |
| H3 | Experiência em dummy (veterano); experiência específica em corridas molhadas; controle Molhada × Idade; efeito fixo piloto × chuva; efeito fixo equipe × chuva; agrupamento duplo (corrida e piloto); sem 2026 |
| H3 complementar | Interações piloto × chuva com teste de Wald (F), variantes com efeito fixo equipe × chuva e com pilotos com ≥ 8 corridas molhadas, e teste de permutação (199 sorteios) |

---

## 6. Principais decisões metodológicas

Todas ficam na seção 0 de cada código e podem ser alteradas ali.

| Decisão | Valor | Onde |
|---|---|---|
| Volta de ritmo | Sem entrada ou saída de box, sem 1ª volta, sem SC, VSC ou bandeira vermelha, e não lenta | `02_tratamento.py` |
| Volta lenta suspeita | > 10% acima da mediana dos outros pilotos nas voltas n±2, no mesmo tipo de pneu | `02_tratamento.py` |
| Corrida molhada | ≥ 10% das voltas com pneu intermediário ou de chuva, **ou** ≥ 10% do tempo com chuva registrada | `02_tratamento.py` |
| Leituras inválidas de clima | Temperatura da pista < 5 °C é tratada como ausente | `02_tratamento.py` |
| Abandono por acidente | Causa oficial em 2021–2022. Nas demais temporadas, aproximação R3 (não recolheu ao box e houve SC, VSC ou bandeira vermelha, abandono na 1ª volta ou mensagem da FIA), validada contra a causa oficial: precisão 68%, cobertura 67% | `02_tratamento.py` |
| GP disputado (experiência) | Largou quando completou ≥ 1 volta. Com 0 voltas, não largou se a Ergast marca posição "W" ou "F", largada não registrada (grid 0) ou status de não largada | `02_tratamento.py` |
| Identificador do piloto ausente | Em 6 corridas, completado pela sigla do piloto | `02_tratamento.py` |
| Winsorização | Percentis 1 e 99 das medidas contínuas de ritmo e consistência | `03_exploracao.py` |
| Taxas por 100 voltas | Apenas pilotos com ≥ 10 voltas na corrida | `04_modelos.py` |
| Medida de experiência (H3) | log(1 + GPs anteriores), centrada na média | `04_modelos.py` |

Os arquivos de `dados/tratados/diagnostico/` documentam as validações: precisão do proxy de acidente
por condição (seca e molhada), classificação dos eventos da direção de prova, origem da informação de
abandono e experiência por piloto.

---

## 7. Saídas: tabelas e figuras

**Exploração (`03_exploracao.py` → `resultados/tabelas/tabelas_exploracao.xlsx`):**

| Tabela | Conteúdo |
|---|---|
| T01–T03 | Composição da amostra, estatísticas descritivas, valores extremos e winsorização |
| T04–T05 | Variação da temperatura, multicolinearidade e temperatura por circuito |
| T06–T07 | Corridas secas e molhadas por piloto; ritmo por quintil de temperatura (H1) |
| T08 | Erros e consistência em corridas secas e molhadas (H2) |
| T09–T10 | Diferenças entre pilotos na chuva e no calor (H3 complementar) |
| T11–T12 | Correlações de Pearson e de Spearman |
| T13 | Indicadores por era do regulamento |
| T14 | Erros e ritmo, seco × molhado, por terço de experiência (H3) |

**Modelos (`04_modelos.py` → `resultados/tabelas/tabelas_modelos.xlsx`):**

| Tabela | Conteúdo |
|---|---|
| M1 | H1: temperatura da pista × ritmo e consistência |
| M2 | H1: modelo no nível da volta |
| M3 | H2: corrida molhada × cada tipo de erro |
| M4 | H3: experiência × efeito da chuva |
| M4b | H3: robustez |
| M4c–M4d | H3 complementar: testes piloto a piloto |
| M5 | Robustez de H1 e H2 |

**Figuras (`resultados/figuras/`):**

| Figuras | Conteúdo |
|---|---|
| F01–F02 | Distribuições e amplitude térmica por circuito |
| F03 | Temperatura × ritmo (H1) |
| F04–F05 | Erros em corridas secas e molhadas, e por faixa de chuva (H2) |
| F06–F07 | Diferenças entre pilotos na chuva e no calor |
| F08 | Correlações |
| F09 | Efeito bruto da chuva por terço de experiência (H3) |
| F10 | Efeito estimado da temperatura (H1) |
| F11 | Efeito estimado da chuva por tipo de erro (H2) |
| F12 | Efeito estimado da chuva conforme a experiência (H3) |
| F13 | Efeito da chuva por piloto (H3 complementar) |

---

## 8. Resultados principais

Coeficientes estimados com efeitos fixos de piloto, circuito e equipe × temporada, e erros agrupados
por corrida. Interpretação e discussão estão no artigo. Os coeficientes indicam associações
condicionais aos efeitos fixos.

| Hipótese | Variável dependente | Coeficiente | EP | p | Tabela |
|---|---|---|---|---|---|
| H1 | Ritmo (% acima da referência), por +1 °C na pista | +0,068 | 0,021 | 0,002 | M1 |
| H1 | Ritmo, nível da volta, por +1 °C | +0,055 | — | < 0,001 | M2 |
| H2 | Rodada ou acidente (0/1), corrida molhada | +0,040 | 0,026 | 0,122 | M3 |
| H2 | Voltas lentas por 100 voltas, corrida molhada | +0,338 | 0,151 | 0,027 | M3 |
| H2 | Inconsistência vs. grid (p.p.), corrida molhada | +0,345 | 0,060 | < 0,001 | M3 |
| H3 | Rodada ou acidente: θ (Molhada × Experiência) | −0,058 | 0,018 | < 0,001 (unilateral) | M4 |
| H3 | Voltas lentas: θ | −0,117 | 0,062 | 0,030 (unilateral) | M4 |

Os resultados completos, com todas as variáveis dependentes e os testes de robustez, estão nas
tabelas M1 a M5.

---

## 9. Limitações conhecidas

- **Abandono por acidente:** de 2023 em diante, é uma aproximação com precisão de 68% e cobertura de
  67% em relação à causa oficial.
- **Poucas corridas molhadas:** são 21 corridas molhadas na amostra principal. Isso reduz a precisão
  das estimativas ligadas à chuva.
- **Experiência e idade:** as duas medidas têm correlação de 0,94, o que limita a separação entre elas.
- **Várias variáveis dependentes:** H2 e H3 testam várias medidas de erro ao mesmo tempo.
- **Dados de 2026:** a temporada estava em andamento na coleta, com um regulamento técnico novo.

---

## 10. Créditos e uso dos dados

- **Dados** obtidos com a biblioteca [FastF1](https://github.com/theOehrly/Fast-F1) (licença MIT) e a
  API [Jolpica F1](https://github.com/jolpica/jolpica-f1), sucessora da Ergast. Este é um trabalho
  acadêmico sem vínculo com a Formula One Group ou a FIA. Os dados são usados apenas para fins
  acadêmicos e não comerciais.
