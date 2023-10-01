namespace NeuralNetDemo.Maths;

public static class MatrixExtensions
{
    public static Matrix Transpose(this Matrix original)
    {
        var transposed = new Matrix(original[0].Count, original.Count);
        for (var i = 0; i < original[0].Count; i++)
        {
            for (var j = 0; j < original.Count; j++)
            {
                transposed[i][j] = original[j][i]; // Fixed the indices here
            }
        }

        return transposed;
    }

    public static Matrix DotProduct(this Matrix matrix, Matrix otherMatrix)
    {
        var rowsA = matrix.Count;
        var colsA = matrix[0].Count;
        var rowsB = otherMatrix.Count;
        var colsB = otherMatrix[0].Count;

        if (colsA != rowsB)
        {
            throw new ArgumentException("Matrix dimensions are not compatible for dot product.");
        }

        var newMatrix = new Matrix(rowsA, colsB);
        for (var i = 0; i < rowsA; i++)
        {
            for (var j = 0; j < colsB; j++)
            {
                var sum = 0.0;
                for (var k = 0; k < colsA; k++)
                {
                    sum += matrix[i][k] * otherMatrix[k][j];
                }

                newMatrix[i][j] = sum;
            }
        }

        return newMatrix;
    }

    // Matrix Addition
    public static Matrix Add(this Matrix matrix, Matrix otherMatrix)
    {
        var newMatrix = new Matrix(matrix.Count, matrix[0].Count);
        if (otherMatrix.Count == 1)
        {
            // do a row-wise addition
            if (matrix.First().Count != otherMatrix.First().Count) throw new ArgumentException();

            for (var rowIndex = 0; rowIndex < matrix.Count; rowIndex++)
            {
                for (var i = 0; i < matrix[0].Count; i++)
                {
                    newMatrix[rowIndex][i] = matrix[rowIndex][i] + otherMatrix[0][i];
                }
            }

            return newMatrix;
        }

        var rows = matrix.Count;
        var cols = matrix[0].Count;

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                newMatrix[i][j] = matrix[i][j] + otherMatrix[i][j];
            }
        }

        return newMatrix;
    }

    public static Matrix Add(this Matrix matrix, double scalar)
    {
        var rows = matrix.Count;
        var cols = matrix[0].Count;

        var newMatrix = new Matrix(matrix.Count, matrix[0].Count);
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                newMatrix[i][j] = matrix[i][j] + scalar;
            }
        }

        return newMatrix;
    }

    // Matrix Subtraction
    public static Matrix Subtract(this Matrix matrix, Matrix otherMatrix)
    {
        if (matrix.Count != otherMatrix.Count || matrix[0].Count != otherMatrix[0].Count)
        {
            throw new ArgumentException("Matrix dimensions are not compatible for subtraction.");
        }

        var rows = matrix.Count;
        var cols = matrix[0].Count;
        var newMatrix = new Matrix(matrix.Count, matrix[0].Count);
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                newMatrix[i][j] = matrix[i][j] - otherMatrix[i][j];
            }
        }

        return newMatrix;
    }

    public static Matrix Subtract(this Matrix matrix, double scalar)
    {
        var rows = matrix.Count;
        var cols = matrix[0].Count;

        var newMatrix = new Matrix(matrix.Count, matrix[0].Count);
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                newMatrix[i][j] = matrix[i][j] - scalar;
            }
        }

        return newMatrix;
    }


    public static Matrix Multiply(this Matrix matrix, Matrix otherMatrix)
    {
        if (matrix.Count != otherMatrix.Count || matrix[0].Count != otherMatrix[0].Count)
        {
            throw new ArgumentException("Matrix dimensions are not compatible for subtraction.");
        }

        var rows = matrix.Count;
        var cols = matrix[0].Count;

        var newMatrix = new Matrix(matrix.Count, matrix[0].Count);
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                newMatrix[i][j] = matrix[i][j] * otherMatrix[i][j];
            }
        }

        return newMatrix;
    }


    public static Matrix Apply(this Matrix matrix, Func<double, double> cellOp)
    {
        var newMatrix = new Matrix(matrix.Count(), matrix[0].Count);
        var rows = matrix.Count;
        var cols = matrix[0].Count;
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                newMatrix[i][j] = cellOp(matrix[i][j]);
            }
        }

        return newMatrix;
    }
}