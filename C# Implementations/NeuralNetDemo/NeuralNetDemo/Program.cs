using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.LossFunctions;
using NeuralNetDemo.Network;

const int batchSize = 10;
const int nIterations = 80_000;
const int hiddenSize = 12;
const double learningRate = 0.0001;
const int nInputFeatures = 1;
const int nOutputFeatures = 1;

var network = new NeuralNetwork(new SumOfSquaresLossFunction());
network.AddDenseLayer(nInputFeatures, hiddenSize, new ReluActivation(), true, "Hidden Weights");
network.AddDenseLayer(nOutputFeatures, null, true, "Output Weights");

network.Train(nIterations, batchSize, learningRate);
network.Train(nIterations, batchSize, learningRate);
Console.WriteLine(network.Predict(5));