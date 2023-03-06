# The Neural Network 101 Series

## [Notebook 1 - Linear functions](./1_linear_models/linear_models.ipynb)
In this notebook, we build an understanding of the mathematical operation that underlie every neural network: the linear transformation.

## [Notebook 2 - A simple neural network](./2_simple_neural_network/simple_neural_network.ipynb)
In this notebook, we use our knowledge of linear combinations to implement a simple neural network from scratch using `numpy`. We introduce the concept of layers and show how layers of linear transformations can be used to approximate linear functions. We also introduce how to load a dataset, batch it, use it to train the network, and other necessities.

## [Notebook 3 - Function Optimization](./3_function_optimization/function_optimization.ipynb)
In this notebook, we take a dive in to function optimization. We consider two ways to think about optimizing a function. First is finding the optimal inputs so as to maximize or minimize the output. Second is to find parameters that minimize the difference between the output of the function and some target value. To do this, we introduce the concept of the gradient and talk about what it is and how to use it.

## [Notebook 4 - Activation Functions and Loss Functions](4_activation_and_loss_functions/4_Activation_units_and_loss_functions.ipynb)
In this notebook, we talk about what it means to approximate a function and how to accomplish this using activation functions and loss functions. The function we intend to approximate depends on the task, and the approximation needs to be posed in the form of an appropriate loss function. To understand this, we analyze the properties of a few activation and loss functions and discuss how they work together to drive training in the right direction.


# Getting Started

NOTE: This is probably out of date by now. Just do what you need to do to get set up to run python, and install the dependencies specified in `environment.yml`



Now for the obsolete instructions:

## Running the notebooks locally

In these tutorials, we will be building and running a neural network in Python as well as using code to illustrate how the components of a neural net work.

We will install and manage Python using [Miniconda](https://conda.io/miniconda.html) and run Python code in [JupyterLab](https://github.com/jupyterlab/jupyterlab) an interactive coding environment. This section will guide you through installing Miniconda and JupyterLab so that you can run the tutorial notebooks on your own computer.

In the root directory:

**1. Download and install Miniconda**

```
curl -s https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh > Miniconda2-latest-MacOSX-x86_64.sh
```

```
source Miniconda2-latest-MacOSX-x86_64.sh
```

By default, Miniconda will be installed to `~/miniconda2/` and this path will be added to your PATH variable.

**2. Create a Conda environment containing Python and other goodies**

```
conda env create -f environment.yml
```

`environment.yml` contains the Python packages that we will be using in the tutorials.

That's it for installation, for now and in future sessions:

**3. Start your environment and run Jupyter Lab**

```
conda activate intelligent-machines
jupyter lab
```

This will launch a browser window with your Jupyter Lab session!

## Run the notebooks on Google Colaboratory

If you'd like to run the notebooks without installing anything on your machine, another option is to open them with [Google Colaboratory](https://colab.research.google.com/) by pasting in the link to a notebook from github