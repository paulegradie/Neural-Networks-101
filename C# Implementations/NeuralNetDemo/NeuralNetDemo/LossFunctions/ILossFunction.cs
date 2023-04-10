namespace NeuralNetDemo.LossFunctions;

public interface ILossFunction
{
    double ComputeLoss(List<List<double>> predictions, List<List<double>> targets);
    List<List<double>> ComputeLossDerivatives(List<List<double>> predictions, List<List<double>> targets);
}