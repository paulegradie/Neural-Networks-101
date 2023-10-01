using NeuralNetDemo.Maths;
using NeuralNetDemo.Network;

namespace NeuralNetDemo.LossFunctions;

public class SumOfSquaresLossFunction : ILossFunction
{
    public double ComputeLoss(Matrix predictions, Matrix targets)
    {
        return predictions
            .Subtract(targets)
            .Apply(x => Math.Pow(x, 2))
            .Select(x => x.Sum())
            .Sum() * 0.5;
    }

    public Matrix ComputeGradient(Matrix predictions, Matrix targets)
    {
        return predictions.Subtract(targets);
    }
}