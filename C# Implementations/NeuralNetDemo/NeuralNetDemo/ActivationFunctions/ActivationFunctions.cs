namespace NeuralNetDemo.ActivationFunctions;

public static class ActivationFunctions
{
    public static double Logistic(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    public static double LogisticDerivative(double x)
    {
        return x * (1 - x);
    }

    public static IActivationFunction CreateRelu()
    {
        return new ReluActivation();
    }

    public static double Relu(double x)
    {
        return Math.Max(0.01 * x, x);
    }

    public static double ReluDerivative(double x)
    {
        return x <= 0 ? 0 : 1;
    }
}