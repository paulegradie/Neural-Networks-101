using System.Diagnostics.CodeAnalysis;
using NeuralNetDemo.Maths;

namespace NeuralNetDemo;

public static class Data
{
    private static readonly Random Rand = new();

    [SuppressMessage("ReSharper.DPA", "DPA0001: Memory allocation issues")]
    public static (Matrix, Matrix) GenerateBatch(int batchSize, int min, int max)
    {
        var batchXs = new Matrix(batchSize, 1, () => (double)Rand.Next(min, max));
        var batchYs = batchXs.Apply(currentVal => Math.Pow(currentVal, 2));
        return (batchXs, batchYs);
    }
}