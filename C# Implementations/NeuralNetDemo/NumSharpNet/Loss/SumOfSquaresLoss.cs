using NumSharp;

namespace NumSharpNet.Loss;

public class SumOfSquaresLoss : ILossFunction
{
    public double ComputeLoss(NDArray predictions, NDArray targets)
    {
        return 0.5 * np.sum(np.power(predictions - targets, 2)).astype(np.float32);
    }

    public NDArray ComputeLossDerivatives(NDArray predictions, NDArray targets)
    {
        return (predictions - targets).astype(np.float32);
    }
}