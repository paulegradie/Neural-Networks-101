using NumSharp;

namespace NumSharpNet.Initialization;

public class GaussianInitializer : IWeightInitializer
{
    private readonly double _mean;
    private readonly double _stdDev;
    public static readonly Random Random = new();

    public GaussianInitializer(double mean, double stdDev)
    {
        _mean = mean;
        _stdDev = stdDev;
    }

    public NDArray Initialize(int nRows, int nCols)
    {
        return np.random.normal(_mean, _stdDev, nRows, nCols).astype(np.float32);
    }
}