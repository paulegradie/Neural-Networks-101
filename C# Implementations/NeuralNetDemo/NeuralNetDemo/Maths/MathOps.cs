namespace NeuralNetDemo.Maths;

public static class MathOps
{
    public static List<List<double>> AddRowWise(this List<List<double>> tensor, List<double> bias)
    {
        var outputs = new List<List<double>>();
        foreach (var row in tensor)
        {
            var newRow = row.Zip(bias).Select(x => x.First + x.Second).ToList();
            outputs.Add(newRow);
        }

        return outputs;
    }

    public static List<List<double>> Transpose(this List<List<double>> original)
    {
        var result = new List<List<double>>();
        for (var i = 0; i < original[0].Count; i++)
        {
            var row = new List<double>();
            for (var j = 0; j < original.Count; j++)
            {
                row.Add(original[j][i]);
            }

            result.Add(row);
        }

        return result;
    }

    public static List<List<double>> Dot(this List<List<double>> tensorA, List<List<double>> tensorB)
    {
        var numRowsA = tensorA.Count;
        var numColsA = tensorA[0].Count;
        var numRowsB = tensorB.Count;
        var numColsB = tensorB[0].Count;
        if (numColsA != numRowsB)
        {
            throw new ArgumentException("Number of columns in matrix A must match number of rows in matrix B.");
        }

        var resultMatrix = new List<List<double>>();
        for (var i = 0; i < numRowsA; i++)
        {
            var newRow = new List<double>();
            for (var j = 0; j < numColsB; j++)
            {
                var sum = 0.0;
                for (var k = 0; k < numColsA; k++)
                {
                    sum += tensorA[i][k] * tensorB[k][j];
                }
                newRow.Add(sum);
                // resultMatrix[i].Add(sum);
            }

            resultMatrix.Add(newRow);
        }

        return resultMatrix;
    }

    public static List<List<double>> ApplyCellForCell(
        this List<List<double>> tensorA,
        List<List<double>> tensorB,
        Func<double, double, double> lambda)
    {
        var outputTensor = new List<List<double>>();
        for (var rowIdx = 0; rowIdx < tensorA.Count; rowIdx++)
        {
            var newRow = new List<double>();
            for (var colIdx = 0; colIdx < tensorA[0].Count; colIdx++)
            {
                var tensorAValue = tensorA[rowIdx][colIdx];
                var tensorBValue = tensorB[rowIdx][colIdx];
                newRow.Add(lambda(tensorAValue, tensorBValue));
            }

            outputTensor.Add(newRow);
        }

        return outputTensor;
    }

    public static List<List<double>> ApplyToAllElements(this List<List<double>> tensor, Func<double, double> lambda)
    {
        var newTensor = new List<List<double>>();
        for (var i = 0; i < tensor.Count; i++)
        {
            var newRow = new List<double>();
            for (var j = 0; j < tensor[0].Count; j++)
            {
                newRow.Add(lambda(tensor[i][j]));
            }

            newTensor.Add(newRow);
        }

        return newTensor;
    }
}