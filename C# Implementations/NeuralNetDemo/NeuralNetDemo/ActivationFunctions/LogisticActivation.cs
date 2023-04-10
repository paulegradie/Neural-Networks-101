namespace NeuralNetDemo.ActivationFunctions;

public class LogisticActivation : IActivationFunction
{
    public double Activate(double x)
    {
        return ActivationFunctions.Logistic(x);
    }

    public double Derivative(double x)
    {
        return ActivationFunctions.LogisticDerivative(x);
    }
}