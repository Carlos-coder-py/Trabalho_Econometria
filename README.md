# Trabalho_Econometria

# O efeito das variáveis climáticas sobre o desempenho e a taxa de erro dos pilotos de Fórmula 1

Trabalho de Econometria I

## Pergunta de pesquisa

Qual é o efeito das variações nas condições climáticas, especificamente a ocorrência de chuva e as
oscilações de temperatura, sobre o desempenho e a taxa de erro dos pilotos de Fórmula 1 durante as
corridas?

| Hipótese | Enunciado resumido | Base de análise |
|---|---|---|
| H1 | A elevação da temperatura piora o ritmo e aumenta a variabilidade do desempenho | volta / piloto-corrida |
| H2 | A chuva aumenta a taxa de erro e a probabilidade de incidentes | piloto-corrida |
| H3 | O efeito do clima é heterogêneo entre os pilotos (interação piloto × clima) | piloto-corrida |

## Dados

- **Fonte:** biblioteca [FastF1](https://docs.fastf1.dev) (dados oficiais de cronometragem da F1,
  estação meteorológica do circuito e mensagens da direção de prova) e API Jolpica/Ergast (resultados).
- **Período:** temporadas 2021 a 2026 (2026 até o GP dos EUA), apenas corridas principais.
- **Estrutura:** dados secundários em painel (piloto × corrida; volta × piloto × corrida).

## Estrutura do repositório

```
├── src/
│   ├── 01_coleta.py        # etapa 1: coleta dos dados brutos (FastF1)
│   ├── 02_tratamento.py    # etapa 2: limpeza e criação das variáveis
│   ├── 03_exploracao.py    # etapa 3: descritivas, correlações, testes e gráficos
│   └── 04_modelos.py       # etapa 4: modelos econométricos (H1, H2 e H3)
├── dados/                  # gerado pelos códigos (não versionado)
│   ├── brutos/             # etapa 1: versão original, exatamente como a FastF1 entrega
│   ├── tratados/           # etapa 2: bases tratadas + diagnostico/
│   └── final/              # etapa 3: bases de análise usadas nos modelos
├── resultados/             # etapa 3 (e 4): tabelas (Excel/CSV) e figuras (PNG)
├── f1_cache/               # cache da FastF1 (não versionado)
├── requirements.txt
└── README.md
```

Cada código é independente: começa por uma seção 0 com os caminhos e os parâmetros (decisões
metodológicas) da etapa e lê apenas o que a etapa anterior gravou em disco. Assim, cada etapa
pode ser executada de novo isoladamente (por exemplo, refazer o tratamento sem coletar de novo).

## Como reproduzir

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/01_coleta.py              # coleta (retoma de onde parou; --offline usa só o cache)
python src/02_tratamento.py          # tratamento: dados/tratados/
python src/03_exploracao.py          # exploração: resultados/ e dados/final/
python src/04_modelos.py             # modelos: tabelas M1–M5 e figuras 9–11
```

A coleta respeita o limite da API (500 chamadas/hora): se ele for atingido, o código para, mantém o
que já foi salvo e continua de onde parou na execução seguinte.

## Principais decisões metodológicas

| Decisão | Valor | Onde |
|---|---|---|
| Volta de ritmo | sem box, sem 1ª volta, sem SC/VSC/bandeira vermelha, até 10% acima da mediana | 02_tratamento.py (seção 0) |
| Corrida molhada | ≥ 10% das voltas com pneu de chuva ou ≥ 10% do tempo com chuva | 02_tratamento.py (seção 0) |
| Medida principal de erro | voltas anuladas + penalidades de pilotagem + rodadas + voltas lentas, por 100 voltas | 02_tratamento.py (seção 0) |
| Abandono por acidente | causa oficial (2021–2022); nas demais, aproximação R3 (não recolheu ao box + SC/VSC/bandeira vermelha, 1ª volta ou mensagem da FIA), validada contra a causa oficial: precisão 68%, cobertura 67% | 02_tratamento.py (seção 0) |
| Exclusão | GP da Bélgica 2021 (sem disputa em bandeira verde) | 02_tratamento.py (seção 0) |
| Winsorização | percentis 1 e 99 das medidas contínuas de ritmo e consistência | 03_exploracao.py (seção 0) |
| Elegibilidade para H3 | ≥ 4 corridas molhadas e ≥ 10 secas | 03_exploracao.py (seção 0) |
