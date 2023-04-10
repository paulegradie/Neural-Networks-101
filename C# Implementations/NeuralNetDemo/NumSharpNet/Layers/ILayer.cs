using NumSharp;
using NumSharpNet.ActivationFunctions;

namespace NumSharpNet.Layers;

public interface ILayer
{
    NDArray ForwardPass(NDArray inputs, bool training);
    void BackwardPass(NDArray derivatives, double learningRate);

    public (int, int) Shape { get; }
    public string Name { get; set; }
    public NDArray Weights { get; }
    public NDArray Inputs { get; set; }
    public NDArray Outputs { get; set; }
    public IActivationFunction? ActivationFunction { get; }
}