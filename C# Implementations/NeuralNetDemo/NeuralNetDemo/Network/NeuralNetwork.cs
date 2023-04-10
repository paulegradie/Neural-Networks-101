using NeuralNetDemo.LossFunctions;

namespace NeuralNetDemo.Network;

public class NeuralNetwork : BaseNetwork
{
    private readonly ILossFunction _lossFunction;

    public NeuralNetwork(ILossFunction lossFunction) : base(lossFunction)
    {
        _lossFunction = lossFunction;
    }
    public void Train(int nIterations, int batchSize, double learningRate)
    {
        for (var iteration = 0; iteration < nIterations; iteration++)
        {
            var (inputs, targets) = Data.GenerateBatch(batchSize);

            var predictions = TrainingForwardPass(inputs);

            if (iteration % 400 == 0)
            {
                var loss = _lossFunction.ComputeLoss(predictions, targets);
                const int val = 7;
                Console.WriteLine($"Current Iteration: {iteration} - Loss: {loss} -- Example Prediction f({val}) = {Predict(val)}");
            }

            BackProp(predictions, targets, learningRate);
        }
    }
}