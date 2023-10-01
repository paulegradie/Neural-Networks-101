using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.LossFunctions;
using NeuralNetDemo.Maths;
using NeuralNetDemo.Network;
using Shouldly;

namespace Tests;

public class Tests
{
    [Fact]
    public void GradientComputationWorks()
    {
        var lossFunc = new SumOfSquaresLossFunction();

        var pred = new Matrix(new Vector(1.0), new Vector(4.0));
        var target = new Matrix(new Vector(2.0), new(3.0));

        var result = lossFunc.ComputeGradient(pred, target);
        result.ShouldBeEquivalentTo(new Matrix(new Vector(-1.0), new(1.0)));
    }

    [Fact]
    public void BackpropShouldWork()
    {
        const int batchSize = 25;
        const double learningRate = 0.0001;

        var network = new NeuralNetwork(new SumOfSquaresLossFunction());
        network.AddDenseLayer(1, 4, new LogisticActivation(), "Hidden");
        network.AddDenseLayer(4, 1, null, "Output");

        network.Layers[0].Weights = new Matrix(new Vector(2, 3, 4, 5));
        network.Layers[1].Weights = new Matrix(new Vector(1.0), new(2.0), new(3.0), new(4.0));
    }


    [Fact]
    public void NetworkPredictsCorrectly()
    {
        var network = new NeuralNetwork(new SumOfSquaresLossFunction());
        network.AddDenseLayer(1, 4, new LogisticActivation(), "Hidden");
        network.AddDenseLayer(4, 1, null, "Output");

        network.Layers[0].Weights = new Matrix(new Vector(2, 3, 4, 5));
        network.Layers[1].Weights = new Matrix(new Vector(1.0), new(2.0), new(3.0), new(4.0));

        var result = network.Predict(3);
        result.ShouldBeGreaterThan(9);
        result.ShouldBeLessThan(10);
    }


    [Fact]
    public void NonLinearsWork()
    {
        ActivationFunctions.Logistic(0).ShouldBe(0.5);
        ActivationFunctions.Logistic(-100).ShouldBe(0, 0.01);
        ActivationFunctions.Logistic(100).ShouldBe(1, 0.01);
    }

    [Fact]
    public void LossFunctionWorks()
    {
        var xs = new Matrix(new Vector(1.0), new(1.0), new(1.0));
        var ys = new Matrix(new Vector(3.0), new(2.0), new(4.0));

        new SumOfSquaresLossFunction().ComputeLoss(xs, ys).ShouldBe(7);
    }

    [Fact]
    public void LogisticDerivWorks()
    {
        ActivationFunctions.LogisticDerivative(5).ShouldBe(-20);
        ActivationFunctions.LogisticDerivative(10).ShouldBe(-90);
    }

    [Fact]
    public void DotProductWorks()
    {
        var tensorA = new Matrix(
            new Vector(1.0, 3.0),
            new(2.0, 4.0),
            new(3.0, 6.0));


        var tensorB = new Matrix(
            new Vector(1, 3, 5, 6),
            new(2, 4, 3, 1));


        tensorA.DotProduct(tensorB)
            .ShouldBeEquivalentTo(new Matrix(
                new Vector(7.0, 15.0, 14.0, 9),
                new(10.0, 22.0, 22.0, 16),
                new(15.0, 33.0, 33.0, 24)
            ));
    }

    [Fact]
    public void TransposeShouldWork()
    {
        var tensorA = new Matrix(
            new Vector(1, 2),
            new(2, 4),
            new(3, 6));

        var tensorB = new Matrix(
            new Vector(1, 2, 3),
            new(2, 4, 6));

        tensorA.Transpose().ShouldBeEquivalentTo(tensorB);
    }

    [Fact]
    public void AddMatricesShouldWorkA()
    {
        var matrixA = new Matrix(
            new Vector(1, 2, 3),
            new Vector(1, 2, 3));

        var matrixB = new Matrix(new Vector(2, 2, 2));

        matrixA.Add(matrixB).ShouldBeEquivalentTo(new Matrix(
            new Vector(3, 4, 5),
            new Vector(3, 4, 5)
        ));
    }


    [Fact]
    public void AddMatricesShouldWorkB()
    {
        var matrixA = new Matrix(
            new Vector(1, 2, 3),
            new Vector(1, 2, 3));

        var matrixB = new Matrix(
            new Vector(2, 3, 2),
            new Vector(7, 3, 3));

        matrixA.Add(matrixB).ShouldBeEquivalentTo(new Matrix(
            new Vector(3, 5, 5),
            new Vector(8, 5, 6)
        ));
    }

    [Fact]
    public void AddScalarShouldWork()
    {
        var matrixA = new Matrix(
            new Vector(1, 2, 3),
            new Vector(1, 2, 3));
        
        matrixA.Add(3).ShouldBeEquivalentTo(new Matrix(
            new Vector(4, 5, 6),
            new Vector(4, 5, 6)
        ));
    }

    [Fact]
    public void SubtractMatricesShouldWorkA()
    {
        var matrixA = new Matrix(
            new Vector(1, 2, 3),
            new Vector(1, 2, 3));

        var matrixB = new Matrix(
            new Vector(2, 2, 2), 
            new Vector(3, 3, 3));

        matrixA.Subtract(matrixB).ShouldBeEquivalentTo(new Matrix(
            new Vector(-1, 0, 1),
            new Vector(-2, -1, 0)
        ));
    }

    [Fact]
    public void SubtractScalarShouldWork()
    {
        var matrixA = new Matrix(
            new Vector(1, 2, 3),
            new Vector(1, -2, 3));

        matrixA.Subtract(-2).ShouldBeEquivalentTo(new Matrix(
            new Vector(3, 4, 5),
            new Vector(3, 0, 5)
        ));
    }

    [Fact]
    public void MultiplyMatricesShouldWorkA()
    {
        var matrixA = new Matrix(
            new Vector(1, 2, 3),
            new Vector(1, 2, 3));

        var matrixB = new Matrix(
            new Vector(2, 2, 2), 
            new Vector(3, 3, 3));

        matrixA.Multiply(matrixB).ShouldBeEquivalentTo(new Matrix(
            new Vector(2, 4, 6),
            new Vector(3, 6, 9)
        ));
    }

    [Fact]
    public void ApplyShouldWork()
    {
        var matrix = new Matrix(
            new Vector(1, 2, 3),
            new Vector(1, 2, 3));

        matrix.Apply(x => x + 4).ShouldBeEquivalentTo(
            new Matrix(
                new Vector(5, 6, 7),
                new Vector(5, 6, 7))
        );
    }
}