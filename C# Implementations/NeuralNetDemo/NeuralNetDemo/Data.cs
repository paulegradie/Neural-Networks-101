using System.Diagnostics.CodeAnalysis;

namespace NeuralNetDemo;

public static class Data
{
    public static readonly Random Rand = new();

    [SuppressMessage("ReSharper.DPA", "DPA0001: Memory allocation issues")]
    public static (List<List<double>>, List<List<double>>) GenerateBatch(int batchSize)
    {
        var batchXs = new List<List<double>>();
        var batchYs = new List<List<double>>();
        for (var i = 0; i < batchSize; i++)
        {
            var val = (double)Rand.Next(-10, 11);
            batchXs.Add(new List<double>() { val });
            batchYs.Add(new List<double>() { Math.Pow(val, 2) });
        }

        return (batchXs, batchYs);
    }
}