namespace NeuralNetDemo.Maths;

public static class WeightInitializer
{
    public static readonly Random Random = new();
    public static double InitializeRandomNormal(double mean, double stdDev)
    {
        var u1 = 1.0 - Random.NextDouble(); // uniform(0,1] random doubles
        var u2 = 1.0 - Random.NextDouble();
        var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // random normal(0,1)
        return mean + stdDev * normal; // random normal(mean, stdDev^2)
    }

    public static double InitializeZero()
    {
        return 0.0;
    }
}