using NeuralNetDemo.Maths;

namespace NeuralNetDemo.LossFunctions;

public interface ILossFunction
{
    double ComputeLoss(Matrix predictions, Matrix targets);
    Matrix ComputeGradient(Matrix predictions, Matrix targets);
}