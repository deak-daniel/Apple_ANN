using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace ANN_V2
{
    /// <summary>
    /// Flags to mark the layers.
    /// </summary>
    public enum NeuronType
    {
        Input = 0,
        Hidden = 1,
        Output = 2
    }
    /// <summary>
    /// The Neural Network class.
    /// </summary>
    [Serializable]
    public class NeuralNetwork
    {
        #region Public properties
        /// <summary>
        /// The Learning Rate parameter, which affects the network's speed of learning.
        /// </summary>
        public readonly static double LearningRate = 0.0005;
        /// <summary>
        /// A boolean parameter, indicating whether the network is trained or not.
        /// </summary>
        public bool IsTrained { get; set; }
        /// <summary>
        /// The first layer, or Input layer. 
        /// </summary>
        public Neuron[] Input_Layer { get; private set; }
        /// <summary>
        /// The hidden layer.
        /// </summary>
        public Neuron[] Hidden_Layer { get; private set; }
        /// <summary>
        /// The output layer, which contains only one neuron, since this is a binary classification problem.
        /// </summary>
        public Neuron[] Output_Layer { get; private set; }
        /// <summary>
        /// The actual output or label.
        /// </summary>
        public double Actual { get; private set; }
        public double[] Labels { get; private set; }
        /// <summary>
        /// The neuron that is chosen by the network.
        /// </summary>
        public Neuron Chosen { get; private set; }
        /// <summary>
        /// The network's predicted output.
        /// </summary>
        public double Predicted { get; private set; }
        /// <summary>
        /// The error's derivative.
        /// </summary>
        public double errorDerivative { get; set; }
        /// <summary>
        /// This value indicates how "wrong" was the network on it's guess.
        /// </summary>
        public double error { get; set; }
        #endregion // Public properties

        #region Contructor
        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="input_layer_number">The number of input parmeters.</param>
        /// <param name="hidden_layer_number">The number of hidden layer neurons (12 by default)</param>
        /// <param name="output_layer_number">The number of neurons in the output layer (1 by default)</param>
        public NeuralNetwork(int input_layer_number, int hidden_layer_number = 12, int output_layer_number = 3)
        {
            Input_Layer = new Neuron[input_layer_number];
            Hidden_Layer = new Neuron[hidden_layer_number];
            Output_Layer = new Neuron[output_layer_number];
            this.error = 1;
            this.IsTrained = false;

            for (int i = 0; i < Hidden_Layer.Length; i++)
            {
                Hidden_Layer[i] = new Neuron(Input_Layer.Length, NeuronType.Hidden);
            }
            for (int i = 0; i < Output_Layer.Length; i++)
            {
                Output_Layer[i] = new Neuron(Hidden_Layer.Length, NeuronType.Output, i);
            }
        }
        #endregion // Constructor

        #region Public methods
        /// <summary>
        /// Initializes the Network, and gives values to the Input layer, preparing the network for forward passing.
        /// </summary>
        /// <param name="inputValues">The list of input values.</param>
        public void Initialize(List<string> inputValues)
        {
            double[] values = inputValues.Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
            if (Input_Layer[0] is not null)
            {
                for (int i = 0; i < inputValues.Count - 1; i++)
                {
                    Input_Layer[i].Activation = Normalize(values, double.Parse(inputValues[i], CultureInfo.InvariantCulture));
                    //Input_Layer[i].Activation = double.Parse(inputValues[i], CultureInfo.InvariantCulture);
                }
            }
            else
            {
                
                for (int i = 0; i < inputValues.Count - 1; i++)
                {
                    Input_Layer[i] = new Neuron(Normalize(values, double.Parse(inputValues[i], CultureInfo.InvariantCulture)));
                    //Input_Layer[i] = new Neuron(double.Parse(inputValues[i], CultureInfo.InvariantCulture));
                }

                for (int i = 0; i < Hidden_Layer.Length; i++)
                {
                    for (int j = 0; j < Input_Layer.Length; j++)
                    {
                        Hidden_Layer[i].Synapses[j] = new Synapse(Input_Layer[j], Hidden_Layer[i]);
                    }
                }

                for (int i = 0; i < Output_Layer.Length; i++)
                {
                    for (int j = 0; j < Hidden_Layer.Length; j++)
                    {
                        Output_Layer[i].Synapses[j] = new Synapse(Hidden_Layer[j], Output_Layer[i]);
                    }
                }
            }

            Actual = double.Parse(inputValues.Last(), CultureInfo.InvariantCulture);

            switch (Actual)
            {
                case 1:
                    Labels = new double[] { 1, 0, 0 };
                    break;
                case 2:
                    Labels = new double[] { 0, 1, 0 };
                    break;
                case 3:
                    Labels = new double[] { 0, 0, 1 };
                    break;
                //case 4:
                //    Labels = new double[] { 0, 0, 0, 1, 0, 0 };
                //    break;
                //case 5:
                //    Labels = new double[] { 0, 0, 0, 0, 1, 0 };
                //    break;
                //case 6:
                //    Labels = new double[] { 0, 0, 0, 0, 0, 1 };
                //    break;

                default:
                    Labels = new double[] { 0, 0, 0 };
                    break;
            }
            
        }
        /// <summary>
        /// Calculating the hidden layer's activation and the output layer's activation.
        /// </summary>
        public void Feedforward()
        {
            foreach (Neuron item in Hidden_Layer)
            {
                foreach (Synapse synapse in item.Synapses)
                {
                    synapse.Initialize();
                }
            }
            
            foreach (Neuron item in Hidden_Layer)
            {
                item.CalculateActivation();
            }
            
            foreach(Neuron item in Output_Layer)
            {
                foreach (Synapse synapse in item.Synapses)
                {
                    synapse.Initialize();
                }
            }

            foreach (Neuron item in Output_Layer)
            {
                item.CalculateActivation();
            }

            Softmax();
            Chosen = Output_Layer.OrderByDescending(x => x.Activation).First();
            Predicted = (double)Chosen?.Activation;
        }
        public static double ReLU(double x)
        {
            return MathF.Max(0, (float)x);
        }
        public static double ReLuDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }
        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }
        public static double TanhDerivative(double x)
        {
            return 1 - (Math.Pow(Tanh(x), 2));
        }
        /// <summary>
        /// Defines the Sigmoid function.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>Sigmoid function's value at input.</returns>
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        /// <summary>
        /// Defines the derivative of the Sigmoid function.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The derivative of the Sigmoid function.</returns>
        public static double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }
        /// <summary>
        /// Error method.
        /// </summary>
        /// <returns>How wrong was the network.</returns>
        public double Error()
        {
            double localError = 0;
            for (int i = 0; i < Output_Layer.Length; i++)
            {
                localError -= (Labels[i] * Math.Log(Output_Layer[i].Activation + 1e-9));
            }
        
            this.error = localError;
            return error;
        }
        /// <summary>
        /// Derivative of the Error method. Used in training the network.
        /// </summary>
        /// <returns>Derivative of the error.</returns>
        public double ErrorDerivative(double? predicted = null)
        {
            double predictedValue = (double)(predicted is null ? Predicted : predicted);
            double errDer = (predictedValue - Actual);
            this.errorDerivative = errDer;
            return errDer;
        }
        /// <summary>
        /// Function that "Selects" the output which is the correct one according to the the network.
        /// </summary>
        public void Softmax()
        {
            double max = Output_Layer.Max(x => x.Activation);
            double[] exps = Output_Layer.Select(x => Math.Exp(x.Activation - max)).ToArray();
            double sumExps = exps.Sum();

            for (int i = 0; i < Output_Layer.Length; i++)
            {
                Output_Layer[i].Activation = exps[i] / sumExps;
            }
        }
        public void Backpropagate()
        {
            for (int z = 0; z < Output_Layer.Length; z++)
            {
                // Compute the derivative of the error with respect to the input of the softmax
                double errorDerivative = Output_Layer[z].Activation - Labels[z];

                // Update the output layer neurons
                Output_Layer[z].CalculateDerivative(errorDerivative);

                for (int i = 0; i < Output_Layer.Length; i++)
                {
                    for (int j = 0; j < Output_Layer[i].Synapses.Length; j++)
                    {
                        Output_Layer[i].Synapses[j].UpdateWeight(errorDerivative);
                    }
                }

                // Update the hidden layer neurons
                for (int i = 0; i < Hidden_Layer.Length; i++)
                {
                    for (int j = 0; j < Hidden_Layer[i].Synapses.Length; j++)
                    {
                        Hidden_Layer[i].Synapses[j].UpdateWeight(errorDerivative);
                    }
                }
            }
        }
        #endregion // Public methods

        #region Private methods
        private double Normalize(double[] values, double value)
        {
            return value / values.Sum();
        }
        #endregion
    }

    /// <summary>
    /// This class is used to connect two <see cref="Neuron"/>
    /// </summary>
    [Serializable]
    public class Synapse
    {
        #region Private fields
        /// <summary>
        /// Random number generator for the weights.
        /// </summary>
        private static Random rnd = new Random();
        #endregion // Private fields

        #region Public properties
        /// <summary>
        /// The first neuron.
        /// </summary>
        public Neuron In { get; set; }
        /// <summary>
        /// The second neuron.
        /// </summary>
        public Neuron Out { get; set; }
        /// <summary>
        /// The weight assigned to the first neuron.
        /// </summary>
        public double Weight { get; set; }
        /// <summary>
        /// The weighted product of the first neuron.
        /// </summary>
        public double WeightedSum { get; set; }
        #endregion // Public properties

        #region Constructor
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="first">The first <see cref="Neuron"/> which to be connected.</param>
        /// <param name="second">The second <see cref="Neuron"/> which to be connected with the first one.</param>
        public Synapse(Neuron first, Neuron second)
        {
            In = first; 
            Out = second;
            Weight = rnd.NextDouble();
        }
        #endregion // Constructor

        #region Public methods
        /// <summary>
        /// The main algorithm behind the network's learning process.
        /// </summary>
        /// <param name="error">The number which represents the network's "wrongness".</param>
        public void UpdateWeight(double error)
        {
            if (In.neuronType == NeuronType.Hidden && Out.neuronType == NeuronType.Output)
            {
                Weight = Weight - (NeuralNetwork.LearningRate * (error * NeuralNetwork.ReLuDerivative(In.Activation) ));
                Out.Bias = Out.Bias - (NeuralNetwork.LearningRate * (error));
            }
            else if (In.neuronType == NeuronType.Input && Out.neuronType == NeuronType.Hidden)
            {
                Weight = Weight - NeuralNetwork.LearningRate * In.Activation * Out.Derivative;
                Out.Bias = Out.Bias - (NeuralNetwork.LearningRate * Out.Derivative);
            }
        }
        /// <summary>
        /// Calculates the weighted product of the input neurom, and then stores it, this is needed to make the 
        /// math a little bit easier in the <see cref="Neuron"/> class.
        /// </summary>
        public void Initialize()
        {
            this.WeightedSum = Weight * In.Activation;
        }
        #endregion // Public methods
    }

    /// <summary>
    /// The class defines a Neuron in the Neural Network.
    /// </summary>
    [Serializable]
    public class Neuron
    {
        #region Private fields
        static Random rnd = new Random();
        #endregion // Private fields

        #region Public properties
        /// <summary>
        /// Property representing the type of the neuron.
        /// </summary>
        public NeuronType neuronType { get; set; }
        /// <summary>
        /// Activation of the Neuron, in the case of the Input Layer, this is the input value.
        /// </summary>
        public double Activation { get; set; }
        /// <summary>
        /// Array of Synapses which defines the connection between this layer, and the next layer to the right.
        /// </summary>
        public Synapse[] Synapses { get; set; }
        /// <summary>
        /// The derivative using the activation function's derivative.
        /// </summary>
        public double Derivative { get; set; }
        /// <summary>
        /// Bias.
        /// </summary>
        public double Bias { get; set; }
        /// <summary>
        /// The id of the neuron.
        /// </summary>
        public int Id { get; private set; }
        #endregion // Public properties

        #region Constructor
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="Input_value_length">The number of neurons connecting to this neuron's containing layer.</param>
        /// <param name="type">The <see cref="NeuronType"/> of the neuron</param>
        public Neuron(int Input_value_length, NeuronType type)
        {
            this.Bias = rnd.NextDouble();
            neuronType = type;
            Synapses = new Synapse[Input_value_length];
        }
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="Input_value_length">The number of neurons connecting to this neuron's containing layer.</param>
        /// <param name="type">The <see cref="NeuronType"/> of the neuron</param>
        /// <param name="id">id</param>
        public Neuron(int Input_value_length, NeuronType type, int id)
        {
            this.Id = id;
            this.Bias = rnd.NextDouble();
            neuronType = type;
            Synapses = new Synapse[Input_value_length];
        }
        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="activation">Used when the neuron type is set to Input.</param>
        public Neuron(double activation)
        {
            neuronType = NeuronType.Input;
            this.Activation = activation;
            this.Bias = rnd.NextDouble();
        }
        #endregion // Constructor

        #region Public methods
        /// <summary>
        /// Calculates the sum of the <see cref="Synapse.WeightedSum"/>
        /// </summary>
        public void CalculateActivation()
        {
            double act = Synapses.Sum(x => x.WeightedSum) + Bias;
            this.Activation = NeuralNetwork.ReLU(act);
        }
        /// <summary>
        /// Calculates the derivative of the activation with respect to the error's derivative
        /// </summary>
        /// <param name="errorDelta">Error's derivative</param>
        public void CalculateDerivative(double errorDelta )
        {
            if (neuronType == NeuronType.Output)
            {
                for (int i = 0; i < this.Synapses.Length; i++)
                {
                    this.Synapses[i].In.Derivative = NeuralNetwork.ReLuDerivative(this.Activation) * this.Synapses[i].Weight * errorDelta;
                    //this.Synapses[i].In.Derivative = this.Activation * this.Synapses[i].Weight * errorDelta;
                }
            }
        }
        #endregion
    }
}
