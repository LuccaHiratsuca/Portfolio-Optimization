module Utils

open System
open System.Threading

/// Annualizes a daily return
let annualizeReturn (dailyMean: float) : float =
    dailyMean * 252.0

/// Annualizes a daily volatility
let annualizeVolatility (dailyStd: float) : float =
    dailyStd * sqrt 252.0

/// Computes annualized Sharpe Ratio
let sharpeRatio (retAnn: float) (volAnn: float) (rf: float) : float =
    (retAnn - rf) / volAnn

/// Thread-safe random number generator
let private rng =
    new ThreadLocal<Random>(fun () ->
        new Random(Environment.TickCount + Thread.CurrentThread.ManagedThreadId))

/// Generates a vector of long-only weights summing to 1, with each weight <= maxPct
let randomWeights (n: int) (maxPct: float) : float[] =
    let r = rng.Value
    let rec loop () =
        let raw = Array.init n (fun _ -> -log (r.NextDouble()))
        let sum = Array.sum raw
        let weights = Array.map (fun x -> x / sum) raw
        if Array.exists (fun x -> x > maxPct) weights then loop ()
        else weights
    loop ()

/// Generates all combinations of k elements from the given array
let combinations (k: int) (arr: 'a[]) : 'a[][] =
    let n = arr.Length
    let rec comb (start: int) (cnt: int) : seq<'a[]> =
        seq {
            if cnt = 0 then
                yield [||]
            else
                for i in start .. (n - cnt) do
                    for rest in comb (i + 1) (cnt - 1) do
                        yield Array.append [| arr.[i] |] rest
        }
    comb 0 k |> Seq.toArray