using NeuralNetDemo.Maths;
using NeuralNetDemo.Network;

namespace NeuralNetDemo.LossFunctions;

public class SumOfSquaresLossFunction : ILossFunction
{
    public static SumOfSquaresLossFunction CreateSumOfSquares()
    {
        return new SumOfSquaresLossFunction();
    }

    public static List<List<double>> SumOfSquaresDerivatives(List<List<double>> predictions, List<List<double>> targets)
    {
        return predictions.ApplyCellForCell(targets, (prediction, target) => prediction - target);
    }

    public static double SumOfSquaresLoss(List<List<double>> predictions, List<List<double>> targets)
    {
        return predictions
            .ApplyCellForCell(targets, (prediction, target) => Math.Pow(prediction - target, 2))
            .SelectMany(x => x)
            .Sum() * 0.5;
    }

    public double ComputeLoss(List<List<double>> predictions, List<List<double>> targets)
    {
        return SumOfSquaresLoss(predictions, targets);
    }

    public List<List<double>> ComputeLossDerivatives(List<List<double>> predictions, List<List<double>> targets)
    {
        return SumOfSquaresDerivatives(predictions, targets);
    }
}