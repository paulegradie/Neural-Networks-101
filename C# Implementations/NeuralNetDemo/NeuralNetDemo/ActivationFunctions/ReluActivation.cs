namespace NeuralNetDemo.ActivationFunctions;

public class ReluActivation : IActivationFunction
{
    public double Activate(double x)
    {
        return ActivationFunctions.Relu(x);
    }

    public double Derivative(double x)
    {
        return ActivationFunctions.ReluDerivative(x);
    }
}