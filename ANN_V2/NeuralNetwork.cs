using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN_V2
{
    public enum NeuronType
    {
        Input = 0,
        Hidden = 1,
        Output = 2
    }
    public class NeuralNetwork
    {
        public readonly static double LearningRate = 0.0005;
        public Neuron[] Input_Layer { get; set; }
        public Neuron[] Hidden_Layer { get; set; }
        public Neuron[] Output_Layer { get; set; }
        public double Actual { get; set; }
        public double error { get; set; }
        public NeuralNetwork(int input_layer_number, int hidden_layer_number = 2, int output_layer_number = 1)
        {
            Input_Layer = new Neuron[input_layer_number];
            Hidden_Layer = new Neuron[hidden_layer_number];
            Output_Layer = new Neuron[output_layer_number];

            for (int i = 0; i < Hidden_Layer.Length; i++)
            {
                Hidden_Layer[i] = new Neuron(Input_Layer.Length, NeuronType.Hidden);
            }
            for (int i = 0; i < Output_Layer.Length; i++)
            {
                Output_Layer[i] = new Neuron(Hidden_Layer.Length, NeuronType.Output);
            }

            
        }
        public void Initialize(List<string> inputValues)
        {
            if (Input_Layer[0] is not null)
            {
                for (int i = 0; i < inputValues.Count; i++)
                {
                    Input_Layer[i].Activation = (double.Parse(inputValues[i], CultureInfo.InvariantCulture));
                }
            }
            else
            {
                for (int i = 0; i < inputValues.Count; i++)
                {
                    Input_Layer[i] = new Neuron(double.Parse(inputValues[i], CultureInfo.InvariantCulture));
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
            
        }
        public void Feedforward()
        {
            for (int i = 0; i < Hidden_Layer.Length; i++)
            {
                for (int j = 0; j < Hidden_Layer[i].Synapses.Length; j++)
                {
                    Hidden_Layer[i].Synapses[j].Initialize();
                }
            }

            for (int i = 0; i < Hidden_Layer.Length; i++)
            {
                Hidden_Layer[i].CalculateActivation();
            }
            

            for (int i = 0; i < Output_Layer.Length; i++)
            {
                for (int j = 0; j < Output_Layer[i].Synapses.Length; j++)
                {
                    Output_Layer[i].Synapses[j].Initialize();
                }
            }
            for (int i = 0; i < Output_Layer.Length; i++)
            {
                Output_Layer[i].CalculateActivation();
            }

            for (int i = 0; i < Output_Layer.Length; i++)
            {
                Output_Layer[i].CalculateDerivative(ErrorDerivative(Output_Layer.First().Activation, Actual));
            }
        }
        public double Error()
        {
            error = Math.Round(Error(Output_Layer.First().Activation, Actual), 2);
            return error;
        }
        private double Error(double prediction, double actual)
        {
            return 0.5 * (Math.Pow((prediction - actual), 2));
        }
        public static double ErrorDerivative(double prediction, double actual)
        {
            return prediction - actual;
        }
        public void Backpropagate()
        {
            double err = ErrorDerivative(Output_Layer.First().Activation, Actual);
            for (int i = 0; i < Output_Layer.Length; i++)
            {
                for (int j = 0; j < Output_Layer[i].Synapses.Length; j++)
                {
                    Output_Layer[i].Synapses[j].UpdateWeight(err);
                }
            }

            for (int i = 0; i < Hidden_Layer.Length; i++)
            {
                for (int j = 0; j < Hidden_Layer[i].Synapses.Length; j++)
                {
                    Hidden_Layer[i].Synapses[j].UpdateWeight(err);
                }
            }


        }
    }
    public class Synapse
    {
        private static Random rnd = new Random();
        public Neuron In { get; set; }
        public Neuron Out { get; set; }
        public double Weight { get; set; }
        public double WeightedSum { get; set; }
        public Synapse(Neuron first, Neuron second)
        {
            In = first; 
            Out = second;
            Weight = rnd.NextDouble();
        }
        public void UpdateWeight(double error)
        {
            if (In.neuronType == NeuronType.Hidden && Out.neuronType == NeuronType.Output)
            {
                Weight = Weight - (NeuralNetwork.LearningRate * (In.Activation * error));
            }
            else if (In.neuronType == NeuronType.Input && Out.neuronType == NeuronType.Hidden)
            {
                Weight = Weight - (NeuralNetwork.LearningRate * In.Activation * Out.Derivative);
            }
        }
        public void Initialize()
        {
            this.WeightedSum = Weight * In.Activation;
        }

    }
    public class Neuron
    {
        public NeuronType neuronType { get; set; }
        public double Activation { get; set; }
        public Synapse[] Synapses { get; set; }
        public double Derivative { get; set; }
        public Neuron(int Input_value_length, NeuronType type)
        {
            neuronType = type;
            Synapses = new Synapse[Input_value_length];
        }
        public Neuron(double activation)
        {
            neuronType = NeuronType.Input;
            this.Activation = activation;
        }
        public void CalculateActivation()
        {
            this.Activation = Synapses.Sum(x => x.WeightedSum);
        }
        public void CalculateDerivative(double errorDelta )
        {
            if (neuronType == NeuronType.Output)
            {
                for (int i = 0; i < this.Synapses.Length; i++)
                {
                    this.Synapses[i].In.Derivative = this.Synapses[i].Weight * errorDelta;
                }
            }
        }
    }
}
