module Simulator

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open Utils
open System
open System.Diagnostics
open FSharp.Collections.ParallelSeq

/// Evaluates the Sharpe Ratio given vectorized returns and covariance
let private simulateSharpe
    (muVec: Vector<float>)
    (cov: Matrix<float>)
    (w: float[])
    (rf: float)
    : float =
    let wVec     = DenseVector.OfArray w
    let dailyRet = muVec.DotProduct wVec
    let dailyVar = wVec.DotProduct (cov * wVec)
    let retAnn   = dailyRet * 252.0
    let volAnn   = sqrt (dailyVar * 252.0)
    sharpeRatio retAnn volAnn rf

/// Simulates nSims long-only portfolios with maxPct constraint
/// Returns the pair (best weights, best Sharpe)
let findBestForStats
    (mu: float[])
    (cov: DenseMatrix)
    (rf: float)
    (nSims: int)
    (maxPct: float)
    : float[] * float =
    let muVec   = DenseVector.OfArray mu
    let nAssets = muVec.Count

    // Generate nSims simulations in parallel and choose the one with the highest Sharpe
    [1 .. nSims]
    |> PSeq.map (fun _ ->
        let w  = randomWeights nAssets maxPct
        let sr = simulateSharpe muVec cov w rf
        w, sr)
    |> PSeq.maxBy snd

/// Runs the series of combinations and measures time
let runSim
    (combos: int[][])
    (evaluateCombo: int[] -> int[] * float[] * float * float)
    (isParallel: bool)
    : (int[] * float[] * float * float)[] * TimeSpan =

    let sw = Stopwatch.StartNew()
    let totalCombos = combos.Length
    let mutable processed = 0

    let results =
        if isParallel then
            combos
            |> PSeq.mapi (fun i combo ->
                processed <- processed + 1
                if processed % 100 = 0 || processed = totalCombos then
                    printfn "Processed %d/%d combinations (parallel)" processed totalCombos
                evaluateCombo combo)
            |> PSeq.toArray
        else
            combos
            |> Array.mapi (fun i combo ->
                processed <- processed + 1
                if processed % 100 = 0 || processed = totalCombos then
                    printfn "Processed %d/%d combinations (sequential)" processed totalCombos
                evaluateCombo combo)
    sw.Stop()
    results, sw.Elapsed