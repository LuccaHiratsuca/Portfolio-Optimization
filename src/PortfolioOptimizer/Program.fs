module PortfolioOptimizer

open System
open System.IO
open System.Diagnostics
open System.Globalization
open System.Threading.Tasks
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Double
open FSharp.Collections.ParallelSeq
open DataLoader
open Simulator
open Utils
open DataFetcherCs
open DotNetEnv

[<EntryPoint>]
let main _ =

    // Carrega variáveis de ambiente
    Env.Load() |> ignore  

    // 0) Cronômetro total
    let totalSw = Stopwatch.StartNew()

    // Parâmetros
    let rf    = 0.02
    let nSims = 1000
    let maxPct = 0.20

    printfn "Starting Portfolio Optimization...\n"

    // 1) Carrega retornos diários até dez/2024
    printfn "1) Loading daily returns up to Dec 2024..."
    let tickers, returns = loadDailyReturns()
    let nAssets = returns.ColumnCount
    printfn "   → Loaded %d assets\n" nAssets

    // 2) Esperança (mu) e covariância (cov)
    printfn "2) Computing expected returns and covariance matrix..."
    let mu = Array.init nAssets (fun j -> Statistics.Mean (returns.Column j))
    let cov =
        Array2D.init nAssets nAssets (fun i j -> Statistics.Covariance(returns.Column i, returns.Column j))
        |> DenseMatrix.OfArray
    printfn "   → Computed mu & cov\n"

    // 3) Gera combinações de 25 ativos
    printfn "3) Generating all combinations of 25 assets..."
    let combos = combinations 25 [|0..nAssets-1|] |> Array.ofSeq
    printfn "   → %d combinations generated\n" combos.Length

    // Função de avaliação de uma combinação
    let evaluateCombo (combo: int[]) : int[] * float[] * float * float =
        let m   = combo.Length
        let muC = combo |> Array.map (fun i -> mu.[i])
        let covC =
            Array2D.init m m (fun i j -> cov.[combo.[i], combo.[j]])
            |> DenseMatrix.OfArray
        let weights, sharpe = Simulator.findBestForStats muC covC rf nSims maxPct
        combo, weights, sharpe, rf

    // Função para rodar simulação paralela
    let runSim (combos: int[][]) (evalFn: int[] -> int[] * float[] * float * float) (useParallel: bool) : (int[] * float[] * float * float)[] * TimeSpan =
        let sw = Stopwatch.StartNew()
        let results: (int[] * float[] * float * float)[] = Array.zeroCreate combos.Length
        if useParallel then
            let options = ParallelOptions(MaxDegreeOfParallelism = Environment.ProcessorCount)
            Parallel.For(0, combos.Length, options, fun i _ -> results.[i] <- evalFn combos.[i]) |> ignore
        else
            for i in 0 .. combos.Length - 1 do
                results.[i] <- evalFn combos.[i]
        sw.Stop()
        results, sw.Elapsed

    // Helpers para benchmarks
    let runSequential () = combos |> Array.map evaluateCombo |> ignore

    let runParallel degreeOfParallelism =
        let results: (int[] * float[] * float * float)[] = Array.zeroCreate (Array.length combos)
        let options = ParallelOptions(MaxDegreeOfParallelism = degreeOfParallelism)
        Parallel.For(0, combos.Length, options, fun i _ -> results.[i] <- evaluateCombo combos.[i]) |> ignore

    let runRuns runs modeName modeFn =
        [ for i in 1..runs do
            printfn "   • %s run %d/%d" modeName i runs
            let sw = Stopwatch.StartNew()
            modeFn()
            sw.Stop()
            yield sw.Elapsed ]

    // 4) Benchmark: sequential, paralelo metade dos cores, paralelo todos os cores
    printfn "4) Running benchmarks:"
    let totalCores = Environment.ProcessorCount
    let halfCores = max 1 (totalCores / 2)
    printfn "   • Detected %d logical processors" totalCores

    printfn "   - Sequential (1 core):"
    let seqTimes = runRuns 5 "Sequential" runSequential

    printfn "   - Parallel (half capacity: %d cores):" halfCores
    let halfParTimes = runRuns 5 "ParallelHalf" (fun () -> runParallel halfCores)

    printfn "   - Parallel (max capacity: %d cores):" totalCores
    let maxParTimes  = runRuns 5 "ParallelMax"  (fun () -> runParallel totalCores)

    // Calcula médias
    let avgMs times = times |> Seq.map (fun (t: TimeSpan) -> t.TotalMilliseconds) |> Seq.average
    let avgSeq  = avgMs seqTimes
    let avgHalf = avgMs halfParTimes
    let avgMax  = avgMs maxParTimes

    printfn "\n   → Seq times: %A" seqTimes
    printfn "   → Half-par times: %A" halfParTimes
    printfn "   → Max-par times: %A" maxParTimes

    printfn "\n   → Average Seq:  %.2f ms" avgSeq
    printfn "   → Average Half: %.2f ms" avgHalf
    printfn "   → Average Max:  %.2f ms\n" avgMax

    // 5) Salvar CSV de comparação de tempos e médias
    let dataDir    = Path.Combine(__SOURCE_DIRECTORY__, "../../data")
    Directory.CreateDirectory dataDir |> ignore

    let runtimesCsv = Path.Combine(dataDir, "runtimes_comparison.csv")
    let runtimesLines =
        [| yield "Mode,Cores,Run,Elapsed_ms"
           for i,t in List.indexed seqTimes  do yield sprintf "Sequential,1,%d,%.0f" (i+1) t.TotalMilliseconds
           for i,t in List.indexed halfParTimes do yield sprintf "ParallelHalf,%d,%d,%.0f" halfCores (i+1) t.TotalMilliseconds
           for i,t in List.indexed maxParTimes  do yield sprintf "ParallelMax,%d,%d,%.0f" totalCores (i+1) t.TotalMilliseconds |]
    File.WriteAllLines(runtimesCsv, runtimesLines)
    printfn "   • Saved runtimes to %s" runtimesCsv

    let avgCsv = Path.Combine(dataDir, "average_runtimes.csv")
    let avgLines =
        [| "Mode,Cores,AverageElapsed_ms"
           sprintf "Sequential,1,%.2f" avgSeq
           sprintf "ParallelHalf,%d,%.2f" halfCores avgHalf
           sprintf "ParallelMax,%d,%.2f" totalCores avgMax |]
    File.WriteAllLines(avgCsv, avgLines)
    printfn "   • Saved average runtimes to %s\n" avgCsv

    // 6) Última simulação paralela: escolhe melhor carteira
    printfn "6) Running final parallel simulation to pick best portfolio..."
    let parResults, _ = runSim combos evaluateCombo true
    let bestIdxs, bestWeights, bestSr, _ = parResults |> Array.maxBy (fun (_,_,sr,_) -> sr)
    let selectedTickers = bestIdxs |> Array.map (fun i -> tickers.[i])
    printfn "   → Best Sharpe = %f\n" bestSr

    // 7) Salvar CSV da melhor carteira
    let portfolioCsv = Path.Combine(dataDir, "best_portfolio.csv")
    let portfolioLines =
        [| yield "Ticker,Weight,Sharpe"
           for i,t in Array.indexed selectedTickers do
             yield sprintf "%s,%.8f,%.6f" t bestWeights.[i] bestSr |]
    File.WriteAllLines(portfolioCsv, portfolioLines)
    printfn "   • Saved best portfolio to %s\n" portfolioCsv

    // 8) Backtest Q1-2025 via Alpha Vantage (C# fetch)
    printfn "8) Fetching Q1-2025 prices and backtesting..."
    let startDate = DateTime(2025,1,1)
    let endDate   = DateTime(2025,4,1)
    let apiKey    = Environment.GetEnvironmentVariable "ALPHAVANTAGE_API_KEY"

    let priceData =
        selectedTickers
        |> Array.map (fun ticker ->
            // Adiciona anotação de tipo ao Task para resolver FS0072
            let fetchTask : Task<System.Collections.Generic.List<System.ValueTuple<DateTime,double>>> =
                DataFetcher.FetchDailyClosesAsync(ticker, startDate, endDate, apiKey)
            let list = fetchTask |> Async.AwaitTask |> Async.RunSynchronously
            let dates  = list |> Seq.map (fun struct(d,_) -> d) |> Seq.toArray
            let closes = list |> Seq.map (fun struct(_,c) -> c) |> Seq.toArray
            ticker, dates, closes)

    // datas comuns
    let commonDates =
        priceData
        |> Array.map (fun (_,ds,_) -> Set.ofArray ds)
        |> Array.reduce Set.intersect
        |> Set.toArray
        |> Array.sort
    let nDays = commonDates.Length - 1
    let nSel  = selectedTickers.Length

    // matriz de retornos diários
    let returnsArr =
        Array2D.init nDays nSel (fun i j ->
            let (_,dates,closes) = priceData.[j]
            let m = dict (Array.zip dates closes)
            let p0 = m.[commonDates.[i]]
            let p1 = m.[commonDates.[i+1]]
            p1 / p0 - 1.0)
    let returnsQ1    = DenseMatrix.OfArray returnsArr
    let portRetDaily = returnsQ1 * DenseVector.OfArray bestWeights

    // 9) Métricas de backtest
    let cumulativeRet = portRetDaily |> Seq.fold (fun acc r -> acc * (1.0+r)) 1.0
    let annRetQ1      = annualizeReturn (Statistics.Mean portRetDaily)
    let volQ1         = annualizeVolatility (Statistics.StandardDeviation portRetDaily)
    let sharpeQ1      = sharpeRatio annRetQ1 volQ1 rf
    printfn "   → Backtest Q1-2025 results:"
    printfn "       Cumulative Return: %.2f%%" ((cumulativeRet - 1.0)*100.0)
    printfn "       Annualized Return: %.2f%%" (annRetQ1*100.0)
    printfn "       Annualized Vol:    %.2f%%" (volQ1*100.0)
    printfn "       Sharpe (rf=%.2f):   %.4f\n" rf sharpeQ1

    // 10) Salvar CSV de métricas do backtest
    let backCsv =
        [| "Metric,Value";
           sprintf "CumulativeReturn,%.6f" cumulativeRet;
           sprintf "AnnualizedReturn,%.6f" annRetQ1;
           sprintf "AnnualizedVolatility,%.6f" volQ1;
           sprintf "Sharpe,%.6f" sharpeQ1 |]
    let backCsvPath = Path.Combine(dataDir, "backtest_q1_2025.csv")
    File.WriteAllLines(backCsvPath, backCsv)
    printfn "   • Saved backtest metrics to %s\n" backCsvPath

    // 11) Salvar CSV de retornos diários do backtest
    let dailyCsvLines =
        [| yield "Date,DailyReturn"
           for i in 1..nDays do
             let d = commonDates.[i]
             let dateStr = sprintf "%04d-%02d-%02d" d.Year d.Month d.Day
             let ret     = portRetDaily.[i-1]
             yield sprintf "%s,%.6f" dateStr ret |]
    let dailyCsvPath = Path.Combine(dataDir, "backtest_q1_2025_daily_returns.csv")
    File.WriteAllLines(dailyCsvPath, dailyCsvLines)
    printfn "   • Saved daily returns to %s\n" dailyCsvPath

    // 12) Tempo total
    totalSw.Stop()
    printfn "Total runtime: %A" totalSw.Elapsed

    0
