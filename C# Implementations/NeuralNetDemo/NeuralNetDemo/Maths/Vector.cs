namespace NeuralNetDemo.Maths;

public class Vector : List<double>
{
    public Vector(int capacity, Func<double>? initializer = null) : base(capacity)
    {
        AddRange(Enumerable.Range(0, capacity).Select(_ => 0.0));
        for (var i = 0; i < capacity; i++)
        {
            this[i] = initializer?.Invoke() ?? this[i];
        }
    }

    public Vector(IEnumerable<double> values)
    {
        AddRange(values);
    }

    public Vector(params double[] values)
    {
        AddRange(values);
    }
}