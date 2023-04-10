using NumSharp;

namespace NumSharpNet.Loss;

public interface ILossFunction
{
    double ComputeLoss(NDArray predictions, NDArray targets);
    NDArray ComputeLossDerivatives(NDArray predictions, NDArray targets);
}