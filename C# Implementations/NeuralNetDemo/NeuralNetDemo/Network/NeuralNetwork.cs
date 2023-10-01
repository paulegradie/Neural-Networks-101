using System.Globalization;
using NeuralNetDemo.LossFunctions;

namespace NeuralNetDemo.Network;

public class NeuralNetwork : BaseNetwork
{
    private readonly ILossFunction _lossFunction;

    public NeuralNetwork(ILossFunction lossFunction) : base(lossFunction)
    {
        _lossFunction = lossFunction;
    }

    public void Train(int batchSize, double learningRate, double targetLoss, int numReportingSteps)
    {
        var loss = 1000.0;
        var step = 0;

        while (loss > targetLoss)
        {
            var (inputs, targets) = Data.GenerateBatch(batchSize, -5, 5);
            var predictions = TrainingForwardPass(inputs);
            BackProp(predictions, targets, learningRate);

            loss = _lossFunction.ComputeLoss(predictions, targets);
            if (step % numReportingSteps == 0)
            {
                Console.WriteLine($"Step: {step}, Loss: {loss.ToString(CultureInfo.InvariantCulture)}, Pred(3): {Predict(3.0)}");
            }

            step += 1;
        }
    }
}