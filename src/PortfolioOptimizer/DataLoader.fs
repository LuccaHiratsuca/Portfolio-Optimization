// DataLoader.fs
module DataLoader

open FSharp.Data
open MathNet.Numerics.LinearAlgebra.Double

let private filePath =
    __SOURCE_DIRECTORY__ + "/../../data/dow_jones_close_prices_aug_dec_2024.csv"

/// Carrega preços de fechamento e retorna tickers + matriz de retornos diários
let loadDailyReturns () : string[] * DenseMatrix =
    let csv     = CsvFile.Load(filePath)
    let headers = csv.Headers.Value
    let tickers = headers |> Array.skip 1

    let prices =
        csv.Rows
        |> Seq.map (fun row ->
            tickers |> Array.map (fun h -> float (row.GetColumn h)))
        |> array2D

    let days   = Array2D.length1 prices
    let assets = Array2D.length2 prices

    let returns =
        Array2D.init (days-1) assets (fun i j ->
            prices.[i+1,j] / prices.[i,j] - 1.0)

    tickers, DenseMatrix.OfArray returns
