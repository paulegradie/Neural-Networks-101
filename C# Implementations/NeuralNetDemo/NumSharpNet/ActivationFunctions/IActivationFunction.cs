using NumSharp;

namespace NumSharpNet.ActivationFunctions;

public interface IActivationFunction
{
    NDArray Activate(NDArray x);
    NDArray Derivative(NDArray x);
}