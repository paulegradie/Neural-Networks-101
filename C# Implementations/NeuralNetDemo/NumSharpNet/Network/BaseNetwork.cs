﻿using NumSharp;
using NumSharpNet.Layers;
using NumSharpNet.Loss;

namespace NumSharpNet.Network;

public abstract class BaseNetwork
{
    private readonly ILossFunction _lossFunction;
    protected readonly List<ILayer> Layers = new();

    protected BaseNetwork(ILossFunction lossFunction)
    {
        _lossFunction = lossFunction;
    }

    public abstract NDArray Predict(NDArray inputs);
    public abstract NDArray TrainingForwardPass(NDArray inputs);

    protected NDArray ForwardPass(NDArray inputs, bool training)
    {
        var x = inputs;
        foreach (var layer in Layers)
        {
            x = layer.ForwardPass(x, training);
        }

        return x;
    }


    protected void BackPropagation(NDArray predictions, NDArray targets, double learningRate)
    {
        var gradients = _lossFunction.ComputeLossDerivatives(predictions, targets);
        for (var i = Layers.Count - 1; i > -1; i--)
        {
            Layers[i].BackwardPass(gradients, learningRate);
            gradients = np.dot(gradients, Layers[i].Weights.T);
            if (i > 0 && Layers[i - 1].ActivationFunction is not null)
            {
                gradients *=  Layers[i - 1].ActivationFunction?.Derivative(Layers[i-1].Outputs);
            }
        }
    }
}