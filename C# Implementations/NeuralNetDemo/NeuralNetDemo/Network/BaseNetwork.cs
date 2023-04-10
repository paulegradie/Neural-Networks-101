using System.Diagnostics.CodeAnalysis;
using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.LossFunctions;
using NeuralNetDemo.Maths;

namespace NeuralNetDemo.Network;

public abstract class BaseNetwork
{
    private readonly ILossFunction _lossFunction;
    protected readonly List<DenseLayer> Layers = new();

    protected BaseNetwork(ILossFunction lossFunction)
    {
        _lossFunction = lossFunction;
    }

    public double Predict(double input)
    {
        var x = new List<List<double>> { new() { input } };
        var prediction = ForwardPass(x, training: false);
        return prediction[0][0];
    }

    protected List<List<double>> TrainingForwardPass(List<List<double>> inputs)
    {
        return ForwardPass(inputs, true);
    }

    private List<List<double>> ForwardPass(List<List<double>> inputs, bool training)
    {
        var x = inputs;
        foreach (var layer in Layers)
        {
            x = layer.ForwardPass(x, training);
        }

        return x;
    }

    public void AddDenseLayer(int nOutputs, IActivationFunction? activationFunction, bool withBias, string? name)
    {
        AddDenseLayer(-1, nOutputs, activationFunction, withBias, name);
    }

    public void AddDenseLayer(int nInputs, int nOutputs, IActivationFunction? activationFunction, bool withBias, string? name)
    {
        if (nInputs <= 0) // if user doesn't specify the row dimension
        {
            if (Layers.Any()) // if there are any previous layers
            {
                // detect cols dim from previous later
                var numPrevLayerCols = Layers.Last().Shape.Item2;
                // set layer dims
                Layers.Add(new DenseLayer(numPrevLayerCols, nOutputs, activationFunction, withBias, name));
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

            Layers.Add(new DenseLayer(nInputs, nOutputs, activationFunction, withBias, name));
        }
    }

    [SuppressMessage("ReSharper.DPA", "DPA0001: Memory allocation issues")]
    protected void BackProp(List<List<double>> predictions, List<List<double>> targets, double learningRate)
    {
        var partialDerivatives = new List<List<double>>();
        for (var i = Layers.Count - 1; i > -1; i--)
        {
            var layer = Layers[i];

            if (i == Layers.Count - 1) // if the last layer
            {
                partialDerivatives = _lossFunction.ComputeLossDerivatives(targets, predictions);
            }
            else
            {
                var previousLayer = Layers[i + 1];
                partialDerivatives = partialDerivatives.Dot(previousLayer.Weights.Transpose());
            }

            partialDerivatives = layer.BackPropPass(partialDerivatives, learningRate);
        }
    }
}