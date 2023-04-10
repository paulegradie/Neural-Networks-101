using Tensorflow;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Losses;
using Tensorflow.NumPy;

// Link together some nodes in a DAG - i.e. assemble the neural net
var inputs = keras.Input(1, dtype: np.float32, name: "x");
var x = keras.layers.Dense(32, activation: keras.activations.Sigmoid, use_bias: true).Apply(inputs);
var outputs = keras.layers.Dense(1, use_bias: true).Apply(x);

// create a model object and compile it
var model = keras.Model(inputs, outputs, name: "xSquared");
model.summary();
model.compile(optimizer: keras.optimizers.SGD((float)0.001), loss: new MeanSquaredError(), metrics: Array.Empty<string>());

// create some data
Random Rand = new();
using var xs = np
    .array(Enumerable.Range(0, 10000).Select(_ => (double)Rand.Next(-10, 11)).ToArray())
    .reshape(new Shape(-1, 1)).astype(np.float32);
using var ys = np.power(xs, 2).astype(np.float32);
model.fit(
    xs,
    ys,
    batch_size: 32,
    epochs: 20);

while (true)
{
    Console.WriteLine();
    Console.WriteLine("Provide a number to square: ");
    var numberString = Console.ReadLine();
    if (numberString == "q")
    {
        break;
    }

    if (double.TryParse(numberString, out var number))
    {
        var result = model.predict(np.array(number).astype(np.float32).reshape(new Shape(1, 1)));
        Console.WriteLine($"{number}^2 = {result}");
    }
}