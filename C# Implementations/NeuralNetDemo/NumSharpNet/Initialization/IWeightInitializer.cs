using NumSharp;

namespace NumSharpNet.Initialization;

public interface IWeightInitializer
{
    NDArray Initialize(int nRows, int nCols);
}