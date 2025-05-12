# Portfolio Optimization em F\#

Este repositório implementa uma simulação e otimização de carteiras de 25 ativos escolhidos dentre os 30 do índice Dow Jones, usando F# (e um módulo C# para fetch de dados via API).


## 1. Arquitetura do Projeto

```bash
Portfolio-Optimization/
├── 📁 data/
│   ├─ 📄 dow_jones_close_prices_aug_dec_2024.csv    # Dados históricos de fechamento
│   ├─ 📄 runtimes_comparison.csv                    # Comparativo de tempos de execução
│   ├─ 📄 average_runtimes.csv                       # Médias de performance por método
│   ├─ 📄 best_portfolio.csv                         # Composição da carteira ótima
│   ├─ 📄 backtest_q1_2025.csv                       # Resultados do backtest (métricas)
│   └─ 📄 backtest_q1_2025_daily_returns.csv         # Retornos diários do backtest
│
├── 📁 scripts/
│   └─ 📄 download_data.py                           # Script Python para download de dados (yfinance)
│
├── 📁 src/
│   ├─ 📁 DataFetcherCs/                             # Projeto C# para coleta de dados
│   │   └─ 📄 DataFetcher.cs                         # Client HTTP para Alpha Vantage API
│   │
│   └─ 📁 PortfolioOptimizer/                        # Núcleo F# da otimização
│       ├─ 📄 .env                                   # Configurações de ambiente
│       ├─ 📄 Utils.fs                               # Funções utilitárias:
│       │   • `annualizeReturn` 📈                  # Annualização de retornos
│       │   • `annualizeVolatility` 📉              # Annualização de volatilidade
│       │   • `sharpeRatio` ⚖️                     # Cálculo do Índice de Sharpe
│       │   • `randomWeights` 🎲                    # Geração de pesos aleatórios
│       │   • `combinations` ➗                      # Combinações de ativos
│       │
│       ├─ 📄 DataLoader.fs                          # Carregamento e transformação de dados:
│       │   • Leitura de CSVs 📂
│       │   • Geração de matriz de retornos 📊
│       │
│       ├─ 📄 Simulator.fs                           # Motor de simulação:
│       │   • `simulateSharpe` 🔄 (μ, cov, w → Sharpe)
│       │   • `findBestForStats` 🏁 (Monte Carlo + maximização)
│       │
│       └─ 📄 Program.fs                             # Orquestração principal:
│           1. Carrega configurações (.env) ⚙️
│           2. Importa dados históricos 📥
│           3. Calcula μ e cov (MathNet) 🧮
│           4. Gera combinações de 25/30 ativos 🔀
│           5. Executa simulação paralela ⚡
│           6. Benchmarks de performance ⏱️
│           7. Backtest Q1-2025 🔄
│           8. Exporta resultados 📤
│
└─ 📄 README.md                                      # Documentação do projeto
```

---

## 2. Como configurar:

Caso queria testar a extração de dados do período de agosto a dezembro de 2024, siga os passo 1 abaixo, caso contrário, pule para o passo 2.

1. **Python & dados históricos**

   * Crie um ambiente virtual Python:

        - MacOS/Linux:
            ```bash
            python3 -m venv venv
            source venv/bin/activate
            ```

        - Windows:
            ```bash
            python -m venv venv
            venv\Scripts\activate
            ```

    * Instale as dependências:
        ```bash
        pip install -r requirements.txt
        ```

    * Execute o script para baixar os dados:
        ```bash
        cd scripts
        python scripts/download_data.py
        ```
        Obs.: O script `download_data.py` baixa os dados históricos de fechamento dos 30 ativos do índice Dow Jones, de agosto a dezembro de 2024, e salva no arquivo `data/dow_jones_close_prices_aug_dec_2024.csv`. O qual já está incluído no repositório.

<h3>Os passos abaixos são para testar ou rodar o projeto:</h3>

2. **Arquivo `.env`**

   Dentro do diretório `src/PortfolioOptimizer`, crie um arquivo `.env` com sua chave AlphaVantage, caso queria encontrar uma chave API, você pode criar uma conta gratuita no site da AlphaVantage e gerar uma chave de API, link: [AlphaVantage](https://www.alphavantage.co/support/#api-key).

    Após ter criado a chave, adicione-a ao arquivo `.env`:

    ```bash
    ALPHAVANTAGE_API_KEY=your_api_key_here
    ```

3. **Compilar & executar**

    Para compilar e executar o projeto, você deve seguir os passos abaixo:

   ```bash
   cd src/PortfolioOptimizer
   dotnet build -c Release
   dotnet run -c Release
   ```

---

## 3. Cálculos principais

Nesta seção irei explicar de forma detalhada cada cálculo essencial do projeto e como ele foi implementado.

### 3.1 Retornos Diários

Para cada ativo *j* e cada par de dias consecutivos (*t-1*, *t*), calculamos o **retorno diário simples**:


$\text{retorno} = \left( \frac{P_t}{P_{t-1}} \right) - 1.0$

**No código:**

```fsharp
let returns =
  Array2D.init (days-1) assets (fun i j ->
    prices.[i+1,j] / prices.[i,j] - 1.0)
```

Onde:
* `prices.[i,j]` é o preço de fechamento do **dia i** para o ativo *j*.
* `Array2D.init (days-1) assets` gera uma matriz de dimensão `(dias-1) × (ativos)`, salvando cada retorno numa posição `[i,j]`.

### 3.2 Sharpe Ratio

O **Sharpe Ratio anualizado** expressa o retorno excedente por unidade de risco:

$\text{Sharpe} = \frac{\text{RetornoAnualizado} - \text{RF}}{\text{VolatilidadeAnualizada}}$

**No código:**

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

O paralelismo é uma técnica usada para tornar mais rápido o processamento de muitas combinações de ativos e simulações. Aqui, explico de forma simples e direta por que ele é usado, como funciona e o que os testes mostram.

### 4.1 Por que usar paralelismo?
Como sabemos que precisamos lidar com:

- **Muitas combinações:** São geradas cerca de 142.506 combinações possíveis a partir de 25 ativos.
- **Simulações extras:** Para cada combinação, fazemos 1.000 simulações de Monte Carlo, criando pesos aleatórios e calculando o Índice de Sharpe.

Como cada combinação e simulação é independente (não precisa esperar as outras), o paralelismo permite dividir o trabalho entre os núcleos do processador, reduzindo o tempo total.

### 4.2 Como o paralelismo é feito

O paralelismo foi implementado de duas formas principais:

1. Em **F#** usando `PSeq` (Parallel Sequence):

    O módulo `PSeq` permite processar sequências de forma paralela, dividindo o trabalho entre os núcleos disponíveis.

    **Exemplo no código:**

    ```fsharp
    combos
    |> PSeq.mapi (fun i combo -> evaluateCombo combo)
    |> PSeq.toArray
    ```

    Isso roda a função `evaluateCombo` em paralelo para cada combinação.

2. Em **C#** usando `Parallel.For`:

    O `Parallel.For` é uma maneira de executar loops em paralelo, dividindo o trabalho entre os núcleos disponíveis.

    **Exemplo no código:**

    ```csharp
    Parallel.For(0, combos.Length, options, fun i _ -> …)
    ```

## Benchmark
Incluí um benchmark no projeto para comparar o desempenho:

- Modo sequencial: Usa só 1 núcleo.
- Paralelo com metade dos núcleos: Usa metade dos núcleos disponíveis.
- Paralelo com todos os núcleos: Usa todos os núcleos.

Rodei cada modo 5 vezes e calculei o tempo médio. O objetivo é mostrar como o tempo diminui quando uso mais núcleos, podemos ver isso na tabela abaixo.

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
Foram implementados os seguintes itens opcionais:

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