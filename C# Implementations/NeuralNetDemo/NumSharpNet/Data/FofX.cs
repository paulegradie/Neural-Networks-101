using NumSharp;

namespace NumSharpNet.Data;

public static class FofX
{
    public static readonly Random Rand = new();

    public static (NDArray, NDArray) GenerateBatch(int batchSize)
    {
        // Match BasicDemo's sampling range. A perfectly symmetric batch distribution
        // around zero can stall hidden-weight learning for this setup.
        var rawBatch = Enumerable.Range(0, batchSize).Select(_ => (double)Rand.Next(-10, 10)).ToArray();

        var batchXs = np.array(rawBatch).reshape(-1, 1).astype(np.float32);
        var batchYs = np.power(batchXs, 2).astype(np.float32);
        return (batchXs, batchYs);
    }
}
