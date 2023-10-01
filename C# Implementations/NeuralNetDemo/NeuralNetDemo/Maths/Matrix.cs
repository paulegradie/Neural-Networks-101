namespace NeuralNetDemo.Maths;

public class Matrix : List<Vector>
{
    public Matrix(int nRows, int nCols, Func<double>? initializer = null)
    {
        foreach (var _ in Enumerable.Range(0, nRows))
        {
            Add(new Vector(nCols, initializer));
        }
    }

    public Matrix(IEnumerable<Vector> rows)
    {
        AddRange(rows);
    }

    public Matrix(params Vector[] rows)
    {
        AddRange(rows);
    }

    public (int, int) GetShape()
    {
        return (Count, this.First().Count);
    }
}