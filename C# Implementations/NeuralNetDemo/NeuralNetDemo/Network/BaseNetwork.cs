using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.LossFunctions;
using NeuralNetDemo.Maths;

namespace NeuralNetDemo.Network;

public abstract class BaseNetwork
{
    private readonly ILossFunction _lossFunction;
    public readonly List<DenseLayer> Layers = new();

    protected BaseNetwork(ILossFunction lossFunction)
    {
        _lossFunction = lossFunction;
    }

    public double Predict(double input)
    {
        var x = new Matrix(1, 1, () => input);
        var prediction = ForwardPass(x, training: false);
        return prediction[0][0];
    }

    protected Matrix TrainingForwardPass(Matrix inputs)
    {
        return ForwardPass(inputs, true);
    }

    private Matrix ForwardPass(Matrix inputs, bool training)
    {
        return Layers
            .Aggregate(inputs, (current, layer) => layer.ForwardPass(current, training));
    }

    public void AddDenseLayer(int nOutputs, IActivationFunction? activationFunction, string? name)
    {
        AddDenseLayer(-1, nOutputs, activationFunction, name);
    }

    public void AddDenseLayer(int nInputs, int nOutputs, IActivationFunction? activationFunction, string? name)
    {
        if (nInputs <= 0) // if user doesn't specify the row dimension
        {
            if (Layers.Any()) // if there are any previous layers
            {
                // detect cols dim from previous later
                var numPrevLayerCols = Layers.Last().Shape.Item2;
                // set layer dims
                Layers.Add(new DenseLayer(numPrevLayerCols, nOutputs, activationFunction, name));
            }
            else // if first layer
            {
                throw new Exception("First layer must set nRows to > 0");
            }
        }
        else // if user provides the row dim
        {
            if (Layers.Any())
            {
                if (Layers.Last().Shape.Item2 != nInputs)
                {
                    throw new Exception($"Incompatible: You're trying to connect ({nInputs}, {nOutputs}) to {Layers.Last().Shape} ");
                }
            }

            Layers.Add(new DenseLayer(nInputs, nOutputs, activationFunction, name));
        }
    }

    protected void BackProp(Matrix predictions, Matrix targets, double lr)
    {
        var gradient = _lossFunction.ComputeGradient(predictions, targets);

        var enumerated = Enumerable.Range(0, Layers.Count).Zip(Layers).ToList();
        enumerated.Reverse();

        foreach (var (i, layer) in enumerated)
        {
            layer.Update(gradient, lr);
            gradient = gradient.DotProduct(layer.Weights.Transpose());

            if (i > 0 && Layers[i - 1].HasActivationFunction)
            {
                gradient = gradient.Multiply(layer.Inputs!.Apply(Layers[i - 1].ActivationFunction!.Derivative));
            }
        }
    }
}