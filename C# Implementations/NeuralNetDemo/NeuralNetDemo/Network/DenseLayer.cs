using System.Diagnostics.CodeAnalysis;
using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.Maths;

namespace NeuralNetDemo.Network;

public class DenseLayer
{
    private readonly IActivationFunction? _activationFunction;
    private readonly bool _withBias;
    public string Name { get; }

    public DenseLayer(int nInput, int nOutput, IActivationFunction? activationFunction, bool? withBias, string? name = "Unnamed Dense Layer", double? mean = 0, double? stdDev = 0.01)
    {
        _activationFunction = activationFunction;
        _withBias = withBias ?? true;
        Name = name ?? string.Empty;
        Weights = new List<List<double>>();
        Biases = new List<double>();
        for (var i = 0; i < nInput; i++)
        {
            var cols = WeightInitializer.Initialize(nOutput, mean ?? 0, stdDev ?? 0.01);
            Weights.Add(cols);
        }

        if (_withBias)
        {
            for (var i = 0; i < nOutput; i++)
            {
                Biases.Add(0.0);
            }
        }
    }

    public List<List<double>> Weights { get; set; }
    public List<double> Biases { get; set; }
    public List<List<double>> Inputs { get; set; } = new();
    public List<List<double>> Outputs { get; set; } = new();

    public (int, int) Shape => GetShape();

    private (int, int) GetShape()
    {
        var nRows = Weights.Count;
        var nCols = Weights[0].Count;
        return (nRows, nCols);
    }


    // layer1 input = 25x1 dot 1x32 hidden
    // layer2 input = 25x32 dot 32x1 output
    [SuppressMessage("ReSharper.DPA", "DPA0001: Memory allocation issues")]
    public List<List<double>> ForwardPass(List<List<double>> batchOfVectors, bool training = true)
    {
        if (batchOfVectors.Any(x => x.Count != GetShape().Item1)) throw new Exception("Matrices don't line up");
        if (training)
        {
            Inputs = batchOfVectors;
        }

        var outputTensor = batchOfVectors.Dot(Weights);
        if (_withBias)
        {
            outputTensor = outputTensor.AddRowWise(Biases);
        }

        if (_activationFunction is not null)
        {
            outputTensor = outputTensor.ApplyToAllElements(_activationFunction.Activate).ToList();
        }

        if (training)
        {
            Outputs = outputTensor;
        }

        return outputTensor;
    }

    [SuppressMessage("ReSharper.DPA", "DPA0001: Memory allocation issues")]
    public List<List<double>> BackPropPass(List<List<double>> partialDerivatives, double learningRate)
    {
        if (_activationFunction is not null)
        {
            var updateTensor = Outputs.ApplyToAllElements(x => _activationFunction.Derivative(x));
            partialDerivatives = partialDerivatives
                .ApplyCellForCell(
                    updateTensor,
                    (partialDerivative, updateVal) => partialDerivative * updateVal);
        }

        // update weights
        var deltas = Inputs
            .Transpose()
            .Dot(partialDerivatives)
            .ApplyToAllElements(x => x * learningRate);
        Weights = Weights.ApplyCellForCell(deltas, (currentWeight, delta) => currentWeight - delta);

        // update bias
        if (Biases.Any())
        {
            var newBiasVector = new List<double>();

            foreach (var (partialDerivRow, bias) in partialDerivatives.Transpose().Zip(Biases))
            {
                var update = bias - partialDerivRow.Sum() * learningRate;
                newBiasVector.Add(update);
            }

            Biases = newBiasVector;
        }

        return partialDerivatives;
    }
}