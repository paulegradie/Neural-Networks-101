using NumSharp;

var lr = 0.0001;
var hidden_size = 16;
var batch_size = 50;
var samples_seen = 0;
var num_iterations = 100000;

var hidden_weights = np.random.normal(0.0, 0.01, 1, hidden_size).astype(np.float32);
var hidden_bias = np.zeros(1, hidden_size).astype(np.float32);

var output_weights = np.random.normal(0.0, 0.01, hidden_size, 1).astype(np.float32);
var output_bias = np.zeros(1, 1).astype(np.float32);
var losses = new List<double>();

for (var _ = 0; _ < num_iterations; _++)
{
    var (xs, ys) = FofX.GenerateBatch(batch_size);
    samples_seen += 1 * batch_size;

    var hidden_layer_out = Logistic.Activate(np.dot(xs, hidden_weights) + hidden_bias);
    var pred = np.dot(hidden_layer_out, output_weights) + output_bias;

    var err = SumOfSquaresLoss.ComputeLoss(pred, ys);
    losses.Add(err);

    var output_derivatives = SumOfSquaresLoss.ComputeLossDerivatives(pred, ys);
    output_weights -= np.dot(hidden_layer_out.T, output_derivatives) * lr;
    output_bias -= np.sum(output_derivatives, axis: 0) * lr;

    var hidden_derivatives = np.dot(output_derivatives, output_weights.T) * Logistic.Derivative(hidden_layer_out);
    hidden_weights -= np.dot(xs.T, hidden_derivatives) * lr;
    hidden_bias -= np.sum(hidden_derivatives, axis: 0) * lr;


    if (_ % 500 == 0)
    {
        Console.WriteLine($"Current batch loss: {err}");
        var inputArray = np.array(4).reshape(1, 1);
        Console.WriteLine($"Going for {4}**2 = {Predictions.Predict(inputArray, hidden_weights, output_weights, hidden_bias, output_bias)[0][0]}");
    }
}

static class Predictions
{
    public static NDArray Predict(NDArray x, NDArray hw, NDArray ow, NDArray hb, NDArray ob)
    {
        var output = Logistic.Activate(np.dot(x, hw) + hb);
        var pred = np.dot(output, ow) + ob;
        return pred;
    }
}


public static class FofX
{
    public static readonly Random Rand = new();

    public static (NDArray, NDArray) GenerateBatch(int batchSize)
    {
        var batchXs = np.random.randint(-10, 10, new Shape(batchSize, 1)).astype(np.float32);
        var batchYs = np.power(batchXs, 2).astype(np.float32);
        return (batchXs, batchYs);
    }
}


public static class SumOfSquaresLoss
{
    public static double ComputeLoss(NDArray predictions, NDArray targets)
    {
        return 0.5 * np.sum(np.power(predictions - targets, 2));
    }

    public static NDArray ComputeLossDerivatives(NDArray predictions, NDArray targets)
    {
        return np.subtract(predictions, targets);
    }
}

public static class Logistic
{
    public static NDArray Activate(NDArray x)
    {
        return 1.0 / (1.0 + np.exp(-x));
    }

    public static NDArray Derivative(NDArray x)
    {
        return x * (1.0 - x);
    }
}