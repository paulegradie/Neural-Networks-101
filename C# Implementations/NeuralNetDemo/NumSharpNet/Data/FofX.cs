using NumSharp;

namespace NumSharpNet.Data;

public static class FofX
{
    public static readonly Random Rand = new();

    public static (NDArray, NDArray) GenerateBatch(int batchSize)
    {
        var batchXs = np.array(new[] { 4, 4, 4, 4, 4, 4, 4, 4 }).reshape(8, 1);
        // var batchXs = np.asarray(Enumerable.Range(0, batchSize).Select(_ => (double)Rand.Next(-10, 11)).ToArray()).reshape(batchSize, 1).astype(np.float32);
        var batchYs = np.power(batchXs, 2).astype(np.float32);
        return (batchXs, batchYs);
    }
}