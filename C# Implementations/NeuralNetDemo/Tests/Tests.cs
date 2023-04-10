using NeuralNetDemo.ActivationFunctions;
using NeuralNetDemo.LossFunctions;
using NeuralNetDemo.Maths;
using Shouldly;

namespace Tests;

public class Tests
{
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
        var xs = new List<List<double>>() { new() { 1 }, new() { 1 }, new() { 1 } };
        var ys = new List<List<double>>() { new() { 3 }, new() { 2 }, new() { 4 } };
        SumOfSquaresLossFunction.SumOfSquaresLoss(xs, ys).ShouldBe(7);
    }

    [Fact]
    public void LogisticDerivWorks()
    {
        ActivationFunctions.LogisticDerivative(5).ShouldBe(-20);
        ActivationFunctions.LogisticDerivative(10).ShouldBe(-90);
    }


    [Fact]
    public void ApplyCellForCellWorks()
    {
        var tensorA = new List<List<double>>()
        {
            new() { 1, 3 },
            new() { 2, 4 },
            new() { 3, 6 }
        };

        var tensorB = new List<List<double>>()
        {
            new() { 1, 3 },
            new() { 2, 4 },
            new() { 3, 6 }
        };

        tensorA
            .ApplyCellForCell(tensorB, (d, d1) => d + d1)
            .ShouldBeEquivalentTo(
                new List<List<double>>()
                {
                    new() { 2, 6, },
                    new() { 4, 8 },
                    new() { 6, 12 }
                }
            );
    }

    [Fact]
    public void ApplyCellForCellOverloadWorks()
    {
        var tensorA = new List<List<double>>()
        {
            new() { 1, 3 },
            new() { 2, 4 },
            new() { 3, 6 }
        };

        tensorA
            .ApplyToAllElements((d) => d * 2)
            .ShouldBeEquivalentTo(
                new List<List<double>>()
                {
                    new() { 2, 6, },
                    new() { 4, 8 },
                    new() { 6, 12 }
                }
            );
    }

    [Fact]
    public void DotProductWorks()
    {
        var tensorA = new List<List<double>>()
        {
            new() { 1, 3 },
            new() { 2, 4 },
            new() { 3, 6 }
        };

        var tensorB = new List<List<double>>()
        {
            new() { 1, 3, 5, 6 },
            new() { 2, 4, 3, 1 },
        };

        tensorA.Dot(tensorB)
            .ShouldBeEquivalentTo(new List<List<double>>()
            {
                new() { 7, 15, 14, 9 },
                new() { 10, 22, 22, 16 },
                new() { 15, 33, 33, 24 },
            });
    }

    [Fact]
    public void TransposeShouldWork()
    {
        var tensorA = new List<List<double>>()
        {
            new() { 1, 3 },
            new() { 2, 4 },
            new() { 3, 6 }
        };

        tensorA.Transpose().ShouldBeEquivalentTo(new List<List<double>>()
        {
            new() { 1, 2, 3 },
            new() { 3, 4, 6 }
        });
    }
}