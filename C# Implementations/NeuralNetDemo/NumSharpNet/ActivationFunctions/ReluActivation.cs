using NumSharp;

namespace NumSharpNet.ActivationFunctions;

public class ReluActivation : IActivationFunction
{
    public NDArray Activate(NDArray x)
    {
        var result = new List<double[]>();
        for (var i = 0; i < x.shape[0]; i++)
        {
            var row = new List<double>();
            for (var j = 0; j < x.shape[1]; j++)
            {
                var update = np.maximum(0.01, x[i, j]);
                row.Add(update);
            }

            result.Add(row.ToArray());
        }

        return np.array(result.ToArray()).astype(np.float32);
    }

    public NDArray Derivative(NDArray x)
    {
        var result = new List<double[]>();
        for (var i = 0; i < x.shape[0]; i++)
        {
            var row = new List<double>();
            for (var j = 0; j < x.shape[1]; j++)
            {
                var res = x[i, j].GetValue<double>() > 0.0 ? 1.0 : 0.0;
                row.Add(res);
            }

            result.Add(row.ToArray());
        }

        return np.array(result.ToArray()).astype(np.float32);
    }
}