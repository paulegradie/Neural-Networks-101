using NumSharp;
using NumSharpNet;
using NumSharpNet.ActivationFunctions;
using NumSharpNet.Loss;

const int batchSize = 10;
const int nIterations = 30_000;
const double learningRate = 0.0001;

var network = new DenseNetwork(new SumOfSquaresLoss());

network.AddDenseLayer(1, 16, new LogisticActivation(),  "Hidden Layer");
network.AddDenseLayer(16, 1, activationFunction: null,  "Output Layer");

network.Train(nIterations, batchSize, learningRate);

var examplePrediction = network.Predict(np.array(2).reshape(1, 1).astype(np.float32));
Console.WriteLine(examplePrediction.GetValue());