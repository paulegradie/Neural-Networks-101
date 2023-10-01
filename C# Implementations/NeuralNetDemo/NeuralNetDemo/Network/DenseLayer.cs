using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.Maths;

namespace NeuralNetDemo.Network;

public class DenseLayer
{
    public readonly IActivationFunction? ActivationFunction;
    public bool HasActivationFunction => ActivationFunction is not null;
    public string Name { get; }

    public DenseLayer(
        int nInput,
        int nOutput,
        IActivationFunction? activationFunction,
        string? name = "Unnamed Dense Layer",
        double? mean = 0,
        double? stdDev = 0.01)
    {
        ActivationFunction = activationFunction;
        Name = name ?? string.Empty;
        Weights = new Matrix(nInput, nOutput, () => WeightInitializer.InitializeRandomNormal(mean ?? 0.0, stdDev ?? 0.01));
        Biases = new Matrix(1, nOutput, WeightInitializer.InitializeZero);
    }

    public Matrix Weights { get; set; }
    public Matrix Biases { get; set; }
    public Matrix? Inputs { get; set; }

    public (int, int) Shape => GetShape();

    private (int, int) GetShape()
    {
        var nRows = Weights.Count;
        var nCols = Weights[0].Count;
        return (nRows, nCols);
    }

    public Matrix ForwardPass(Matrix inputs, bool training = true)
    {
        if (inputs.Any(x => x.Count != GetShape().Item1)) throw new Exception("Matrices don't line up");
        if (training)
        {
            Inputs = inputs;
        }

        var linearCombination = inputs.DotProduct(Weights).Add(Biases);

        return ActivationFunction is null
            ? linearCombination
            : linearCombination.Apply(ActivationFunction.Activate);
    }

    public void Update(Matrix gradient, double lr)
    {
        if (Inputs is null) throw new Exception("Inputs weren't set during training");

        var deltas = Inputs.Transpose().DotProduct(gradient).Apply((val) => val * lr);
        Weights = Weights.Subtract(deltas);

        var biasDeltas = gradient.Sum(x => x.Sum()) * lr;
        Biases = Biases.Subtract(biasDeltas);
    }
}