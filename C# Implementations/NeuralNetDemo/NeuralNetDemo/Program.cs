using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.LossFunctions;
using NeuralNetDemo.Network;

const int batchSize = 30;
const double learningRate = 0.001;

var network = new NeuralNetwork(new SumOfSquaresLossFunction());
network.AddDenseLayer(1, 32, new LogisticActivation(), "Hidden");
network.AddDenseLayer(32, 1, null, "Output");

network.Train(batchSize, learningRate, 5.0, 5000);
Console.WriteLine(network.Predict(5));