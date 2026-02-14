using NumSharp;

namespace NumSharpNet.ActivationFunctions;

public class LogisticActivation : IActivationFunction
{
    public NDArray Activate(NDArray x)
    {
        return 1.0 / (1.0 + np.exp(-x));
    }

    public NDArray Derivative(NDArray x)
    {
        return np.multiply(x, np.subtract(np.array(1.0), x)); 
    }
}