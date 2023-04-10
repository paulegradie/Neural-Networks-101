using NumSharp;
using NumSharpNet.ActivationFunctions;

namespace NumSharpNet.Layers;

public class DenseLayer : ILayer
{
    public IActivationFunction? ActivationFunction { get; }

    public DenseLayer(int nInput, int nOutput, IActivationFunction? activationFunction, string? name = "Unnamed Layer")
    {
        Name = name ?? string.Empty;
        Weights = np.random.normal(0.0, 0.01, nInput, nOutput).astype(np.float32);
        Bias = np.zeros(1, nOutput).astype(np.float32);
        ActivationFunction = activationFunction;
    }

    public NDArray Weights { get; set; }
    public NDArray? Bias { get; set; }
    public string Name { get; set; }
    public (int, int) Shape => (Weights.Shape[0], Weights.Shape[1]);

    public NDArray? Inputs { get; set; }
    public NDArray? Outputs { get; set; }

    public NDArray ForwardPass(NDArray inputs, bool training)
    {
        var output = inputs.dot(Weights) + Bias;
        if (ActivationFunction is not null)
        {
            output = ActivationFunction.Activate(output);
        }

        if (training)
        {
            Inputs = inputs;
            Outputs = output;
        }

        return output;
    }

    public void BackwardPass(NDArray derivatives, double learningRate)
    {
        if (Inputs is null || Outputs is null) throw new Exception("didn't track inputs and outputs during training forward pass");
        Weights -= Inputs.T.dot(derivatives) * learningRate;
        Bias -= np.sum(derivatives, 0) * learningRate;
    }
}