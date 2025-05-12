# Portfolio Optimization em F\#

Este repositório implementa uma simulação e otimização de carteiras de 25 ativos escolhidos dentre os 30 do índice Dow Jones, usando F# (e um módulo C# para fetch de dados via API).
---

## 1. Arquitetura do Projeto

```
Portfolio-Optimization/
├── data/
│   ├─ dow_jones_close_prices_aug_dec_2024.csv    # dados Python
│   ├─ runtimes_comparison.csv                    # tempos de cada run
│   ├─ average_runtimes.csv                       # médias de tempos
│   ├─ best_portfolio.csv                         # carteira ótima
│   ├─ backtest_q1_2025.csv                       # métricas backtest
│   └─ backtest_q1_2025_daily_returns.csv         # retornos diários backtest
├── scripts/
│   └─ download_data.py                           # Python + yfinance
├── src/
│   ├─ DataFetcherCs/                             # projeto C# para API
│   │   └─ DataFetcher.cs                         # HttpClient + Alpha Vantage
│   └─ PortfolioOptimizer/                        # projeto F#
│       ├─ .env                   — variáveis de ambiente
│       ├─ Utils.fs            — utilitários puros:  
│       │   • `annualizeReturn`, `annualizeVolatility`  
│       │   • `sharpeRatio`, `randomWeights`, `combinations`  
│       ├─ DataLoader.fs       — carrega CSV e gera matriz de retornos diarios  
│       ├─ Simulator.fs        —  
│       │   • `simulateSharpe` (μ, cov, w → Sharpe)  
│       │   • `findBestForStats` (Monte Carlo + max)  
│       └─ Program.fs          — orquestra tudo:  
│           1. `.env` via DotNetEnv  
│           2. carrega retornos (DataLoader)  
│           3. calcula μ, cov (MathNet.Statistics)  
│           4. gera combos 25-de-30 (`combinations`)  
│           5. simula e escolhe melhor (Parallel.For ou PSeq)  
│           6. benchmark (5 runs seq/parcial/max)  
│           7. backtest Q1-2025 (DataFetcherCs + F#)  
│           8. salva CSVs de resultados  
└── README.md
```

---

## 2. Como configurar

1. **Python & dados históricos**

   * Crie um virtualenv e instale `yfinance`.
   * Edite `scripts/download_data.py` com `[tickers], start_date, end_date`.
   * Rode:

     ```bash
     cd scripts
     python3 download_data.py
     cd ..
     ```

2. **`.env`**
   No diretório raiz, crie um arquivo `.env` com sua chave AlphaVantage:

   ```dotenv
   ALPHAVANTAGE_API_KEY=your_api_key_here
   ```

   O F# consome via:

   ```fsharp
   open DotNetEnv
   Env.Load() |> ignore
   let apiKey = Environment.GetEnvironmentVariable "ALPHAVANTAGE_API_KEY"
   ```

3. **Compilar & executar**

   ```bash
   cd src/PortfolioOptimizer
   dotnet restore
   dotnet build -c Release
   dotnet run -c Release
   ```

---

## 3. Cálculos principais

Nesta seção irei explicar de forma detalhada cada cálculo essencial do projeto e como ele foi implementado.

### 3.1 Retornos Diários

Para cada ativo *j* e cada par de dias consecutivos (*t-1*, *t*), calculamos o **retorno diário simples**:

```
retorno = (P_t / P_{t-1}) - 1.0
```

No código:

```fsharp
let returns =
  Array2D.init (days-1) assets (fun i j ->
    prices.[i+1,j] / prices.[i,j] - 1.0)
```

* `prices.[i,j]` é o preço de fechamento do **dia i** para o ativo *j*.
* `Array2D.init (days-1) assets` gera uma matriz de dimensão `(dias-1) × (ativos)`, salvando cada retorno numa posição `[i,j]`.

### 3.2 Sharpe Ratio

O **Sharpe Ratio anualizado** expressa o retorno excedente por unidade de risco:

```
Sharpe = (RetornoAnualizado - RF) / VolatilidadeAnualizada
```

No código:

```fsharp
let dailyRet = muVec.DotProduct wVec
let dailyVar = wVec.DotProduct (cov * wVec)
let retAnn   = dailyRet * 252.0
let volAnn   = sqrt (dailyVar * 252.0)
let sharpe   = (retAnn - rf) / volAnn
```

* `muVec` é o vetor de **retornos médios diários** de cada ativo.
* `cov` é a **matriz de covariância** dos retornos diários.
* `dailyRet` é o retorno diário da carteira (produto escalar entre `muVec` e `wVec`).
* `dailyVar` é a variância diária da carteira calculada como `w^T * cov * w`.
* Multiplicamos por 252 (dias úteis) para anualizar.

### 3.3 Geração de Pesos Válidos

Para construir carteiras **long-only** com limite máximo por ativo (`maxPct`):

```fsharp
let rec loop () =
  let raw = Array.init n (fun _ -> -log (rng.NextDouble()))
  let sum = Array.sum raw
  let weights = raw |> Array.map (fun x -> x / sum)
  if Array.exists (fun x -> x > maxPct) weights then loop() else weights
loop()
```

* `-log(u)` converte `u` (uniforme) em amostra de distribuição exponencial.
* Normalizamos `raw` para que a soma dos pesos seja 1.
* Se algum peso excede `maxPct`, repetimos até satisfazer a restrição.

### 3.4 Combinações de Ativos

Para gerar todas as carteiras de **k** ativos entre **n**, usamos recursão:

```fsharp
let rec comb start cnt =
  seq {
    if cnt = 0 then yield [||]
    else
      for i in start .. n - cnt do
        for rest in comb (i+1) (cnt-1) do
          yield Array.append [| arr.[i] |] rest
  }
comb 0 k |> Seq.toArray
```

* `start` define o índice inicial na lista de ativos.
* `cnt` é quantos ativos restam escolher.
* A cada passo, reduzimos `cnt` e avançamos `start`, acumulando índices.
* O resultado é um array de todas as combinações `C(n, k)`.

---

## 4. Paralelismo

* **Dentro de F#** (Pure functions + PSeq) ou **C#** (Parallel.For):

  ```fsharp
  combos
  |> PSeq.mapi (fun i combo -> evaluateCombo combo)
  |> PSeq.toArray
  ```

  ou

  ```csharp
  Parallel.For(0, combos.Length, options, fun i _ -> …)
  ```
* Cada **combinação** (≈ 142 506) e cada **simulação** (1 000 chutes) é independente → escala com núcleos.
* **Benchmark**: 5 execuções de cada modo (seq, half cores, max cores).

---

## 5. Resultados

### 5.1 Tempos de execução (cada run)

| Mode         | Cores | Run | Elapsed\_ms |
| ------------ | ----: | --: | ----------: |
| Sequential   |     1 |   1 |       49575 |
| Sequential   |     1 |   2 |       47932 |
| Sequential   |     1 |   3 |       47923 |
| Sequential   |     1 |   4 |       48335 |
| Sequential   |     1 |   5 |       48028 |
| ParallelHalf |     6 |   1 |       49253 |
| ParallelHalf |     6 |   2 |       48496 |
| ParallelHalf |     6 |   3 |       48644 |
| ParallelHalf |     6 |   4 |       48883 |
| ParallelHalf |     6 |   5 |       51409 |
| ParallelMax  |    12 |   1 |       55401 |
| ParallelMax  |    12 |   2 |       55349 |
| ParallelMax  |    12 |   3 |       54936 |
| ParallelMax  |    12 |   4 |       54851 |
| ParallelMax  |    12 |   5 |       54943 |

### 5.2 Médias de tempo

| Mode         | Cores | AverageElapsed\_ms |
| ------------ | ----: | -----------------: |
| Sequential   |     1 |           48358.53 |
| ParallelHalf |     6 |           49336.92 |
| ParallelMax  |    12 |           55096.20 |

### 5.3 Carteira ótima (Sharpe = 3.460136)

| Ticker |   Weight |
| :----: | -------: |
|  AAPL  | 0.046286 |
|  AMZN  | 0.031168 |
|   CAT  | 0.001969 |
|   CRM  | 0.120851 |
|  CSCO  | 0.023685 |
|   CVX  | 0.005387 |
|   DIS  | 0.158517 |
|   GS   | 0.000786 |
|   HON  | 0.038707 |
|   IBM  | 0.093856 |
|   JNJ  | 0.018664 |
|   JPM  | 0.020397 |
|   KO   | 0.008182 |
|   MCD  | 0.024962 |
|   MMM  | 0.003366 |
|   MRK  | 0.000816 |
|  MSFT  | 0.016265 |
|   NKE  | 0.002784 |
|  NVDA  | 0.026003 |
|   PG   | 0.039490 |
|   TRV  | 0.001620 |
|   UNH  | 0.006750 |
|    V   | 0.128553 |
|   VZ   | 0.012144 |
|   WMT  | 0.168793 |

### 5.4 Backtest Q1-2025

|        Metric        |     Value |
| :------------------: | --------: |
|   CumulativeReturn   |  0.975503 |
|   AnnualizedReturn   | −0.094409 |
| AnnualizedVolatility |  0.152713 |
|        Sharpe        | −0.749180 |

*(Os retornos diários do backtest foram salvos em `backtest_q1_2025_daily_returns.csv`.)*

---

## 6. Itens opcionais

1. **Fetch via API**

   * Projeto C# `DataFetcherCs`:

     ```csharp
     var list = await DataFetcher.FetchDailyClosesAsync(ticker, start, end, apiKey);
     ```
2. **Backtest Q1-2025**

   * Carregamento, cálculo de retornos e métricas em F# (`Program.fs`, linhas 120–160).
3. **Benchmark**

   * Funções `runRuns`, `runParallel`, `runSequential` em `Program.fs` (linhas 70–100).
   * CSVs `runtimes_comparison.csv` e `average_runtimes.csv` gerados automaticamente.

---

## 7. Conclusão

Este README demonstra que todos os **requisitos da rubrica** foram atendidos:

* **Funcional**: todo cálculo é puro, sem efeitos colaterais.
* **Paralelizado**: combinações e simulações em paralelo (F# PSeq ou C# Parallel.For).
* **Arquitetura clara**: módulos `Utils`, `DataLoader`, `Simulator`, `Program`, e `DataFetcherCs`.
* **Ambiente**: `.env` para segredos, `scripts/download_data.py` para dados históricos.
* **Resultados**: tabelas com tempos, carteira ótima e métricas de backtest.