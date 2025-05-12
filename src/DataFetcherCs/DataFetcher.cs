using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace DataFetcherCs
{
    public static class DataFetcher
    {
        private static readonly HttpClient _http = new HttpClient();

        public static async Task<List<(DateTime Date, double Close)>> FetchDailyClosesAsync(
            string ticker,
            DateTime startDate,
            DateTime endDate,
            string apiKey)
        {
            var url = $"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={apiKey}";
            using var resp = await _http.GetAsync(url);
            resp.EnsureSuccessStatusCode();

            using var doc = await JsonDocument.ParseAsync(await resp.Content.ReadAsStreamAsync());
            var root = doc.RootElement;

            if (root.TryGetProperty("Note", out var note))
                throw new Exception($"AlphaVantage note: {note.GetString()}");
            if (root.TryGetProperty("Error Message", out var err))
                throw new Exception($"AlphaVantage error: {err.GetString()}");

            if (!root.TryGetProperty("Time Series (Daily)", out var ts))
                throw new Exception($"Ticker '{ticker}': resposta JSON não contém 'Time Series (Daily)'");

            var results = new List<(DateTime Date, double Close)>();
            foreach (var property in ts.EnumerateObject())
            {
                if (!DateTime.TryParse(property.Name, out var date))
                    continue;
                if (date < startDate || date >= endDate)
                    continue;

                var fields = property.Value;
                if (fields.TryGetProperty("4. close", out var closeProp))
                {
                    var closeStr = closeProp.ValueKind == JsonValueKind.String
                        ? closeProp.GetString()
                        : closeProp.GetRawText();
                    if (double.TryParse(closeStr, out var close))
                        results.Add((date, close));
                }
            }

            // Ordena por data ascendente (Item1 é a Date)
            results.Sort((a, b) => a.Item1.CompareTo(b.Item1));
            return results;
        }
    }
}
