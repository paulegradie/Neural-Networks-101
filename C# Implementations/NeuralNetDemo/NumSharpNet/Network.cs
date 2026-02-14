using NumSharp;
using NumSharpNet.ActivationFunctions;
using NumSharpNet.Data;
using NumSharpNet.Layers;
using NumSharpNet.Loss;
using NumSharpNet.Network;

namespace NumSharpNet;

public class DenseNetwork : BaseNetwork
{
    private readonly ILossFunction _lossFunction;

    public DenseNetwork(ILossFunction lossFunction) : base(lossFunction)
    {
        _lossFunction = lossFunction;
    }

    public override NDArray Predict(NDArray inputs)
    {
        return ForwardPass(inputs, training: false);
    }

    public override NDArray TrainingForwardPass(NDArray inputs)
    {
        return ForwardPass(inputs, true);
    }

    public void Train(int nIterations, int batchSize, double learningRate)
    {
        for (var iteration = 0; iteration < nIterations; iteration++)
        {
            var (inputs, targets) = FofX.GenerateBatch(batchSize);

            var predictions = TrainingForwardPass(inputs);

            if (iteration != 0 && iteration % 600 == 0)
            {
                var loss = np.sum(_lossFunction.ComputeLoss(predictions, targets));
                Console.WriteLine($"Current Iteration: {iteration} - Loss: {loss}");
            }

            BackPropagation(predictions, targets, learningRate);
        }

        var testValue = new[] { -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 };
        var val = np.asarray(testValue).reshape(-1, 1).astype(np.float32);
        var pred = Predict(val);
        var cnt = 0;
        foreach (var v in pred.AsIterator())
        {
            Console.WriteLine($"Example Prediction f({testValue[cnt]}) = {v.ToString()}");
            cnt++;
        }
    }

    public void AddDenseLayer(int? nInputs, int nOutputs, IActivationFunction? activationFunction, string? name = "Unnamed")
    {
        if (nInputs is null && Layers.Count == 0)
        {
            throw new Exception("nInputs cannot be determined or inferred");
        }

        if (nInputs is null)
        {
            nInputs = Layers.Last().Shape.Item2;
        }

        var inputDims = nInputs ?? throw new Exception("Unable to set inputDims");

        Layers.Add(new DenseLayer(inputDims, nOutputs, activationFunction, name));
    }
}