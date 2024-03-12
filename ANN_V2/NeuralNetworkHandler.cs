using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Linq;

namespace ANN_V2
{
    public class NeuralNetworkReader
    {
        public NeuralNetworkReader()
        {
        }
        public NeuralNetwork Read(string path)
        {
            int input_layer_length = 0;
            XmlReader reader1 = XmlReader.Create(path);
            while (reader1.Read())
            {
                
                if (reader1.NodeType == XmlNodeType.Element)
                {
                    if (reader1.Name.Contains("inputneuron_activation"))
                    {
                        input_layer_length++;
                    }
                }
            }


            XmlReader reader2 = XmlReader.Create(path);
            NeuralNetwork network = new NeuralNetwork(input_layer_length);
            network.Initialize();

            int index = 0;
            int index2 = 0;
            string state = "";
            while (reader2.Read())
            {
                if (reader2.NodeType == XmlNodeType.Element)
                {
                    if (reader2.Name.Contains("hiddenneuron_activation"))
                    {
                        string[] helper = reader2.Name.Split('_');
                        index = int.Parse(helper.Last());
                        state = "hidden";
                    }
                    else if (reader2.Name.Contains("hiddenneuron_derivative"))
                    {
                        string[] helper = reader2.Name.Split('_');
                        index = int.Parse(helper.Last());
                        state = "hidden_derivative";
                    }
                    else if (reader2.Name.Contains("outputneuron"))
                    {
                        string[] helper = reader2.Name.Split('_');
                        index = int.Parse(helper.Last());
                        state = "output";
                    }
                    else if (reader2.Name.Contains("outputneuron") && reader2.Name.Contains("synapse"))
                    {
                        string[] helper = reader2.Name.Split('_');
                        index = int.Parse(helper[1]);
                        index2 = int.Parse(helper[3]);
                        state = "output_synapse";
                    }
                    else if (reader2.Name.Contains("hiddenneuron") && reader2.Name.Contains("synapse"))
                    {
                        string[] helper = reader2.Name.Split('_');
                        index = int.Parse(helper[1]);
                        index2 = int.Parse(helper[3]);
                        state = "hidden_synapse";
                    }
                }
                else if (reader2.NodeType == XmlNodeType.Text)
                {
                    if (state == "hidden")
                    {
                        double val = double.Parse(reader2.Value, CultureInfo.InvariantCulture);
                        network.Hidden_Layer[index].Activation = val;
                    }
                    else if (state == "hidden_derivative")
                    {
                        double val = double.Parse(reader2.Value, CultureInfo.InvariantCulture);
                        network.Hidden_Layer[index].Derivative = val;
                    }
                    else if (state == "output")
                    {
                        double val = double.Parse(reader2.Value, CultureInfo.InvariantCulture);
                        network.Output_Layer[index].Activation = val;
                    }
                    else if (state == "hidden_synapse")
                    {
                        double val = double.Parse(reader2.Value, CultureInfo.InvariantCulture);
                        network.Output_Layer[index].Synapses[index2].Weight = val;
                    }
                }
            }

            return network;
        }
    }
    public class NeuralNetworkWriter
    {
        private NeuralNetwork network;
        public NeuralNetworkWriter(NeuralNetwork network)
        {
            this.network = network;
        }
        public void Write(string path)
        {
            XmlWriter writer = XmlWriter.Create(path);
            

            writer.WriteStartElement("NeuralNetwork");

            PropertyInfo inputProps = network.GetType().GetProperty("Input_Layer"); //Gets the properties
            IList input = inputProps.GetValue(network, null) as IList;

            for (int i = 0; i < input.Count; i++)
            {
                writer.WriteElementString($"inputneuron_activation_{i}", (input[i] as Neuron).Activation.ToString());
            }

            PropertyInfo hiddenProps = network.GetType().GetProperty("Hidden_Layer"); //Gets the properties
            IList hidden = hiddenProps.GetValue(network, null) as IList;
            
            for (int i = 0; i < hidden.Count; i++)
            {
                writer.WriteElementString($"hiddenneuron_activation_{i}", (hidden[i] as Neuron).Activation.ToString());
                for (int j = 0; j < (hidden[i] as Neuron).Synapses.Length; j++)
                {
                    writer.WriteElementString($"hiddenneuron_{i}_synapse_{j}_weight", (hidden[i] as Neuron).Synapses[j].Weight.ToString());

                }
            }
            for (int i = 0; i < hidden.Count; i++)
            {
                writer.WriteElementString($"hiddenneuron_derivative_{i}", (hidden[i] as Neuron).Derivative.ToString());
            }

            PropertyInfo outputProps = network.GetType().GetProperty("Output_Layer"); //Gets the properties
            IList output = outputProps.GetValue(network, null) as IList;

            for (int i = 0; i < output.Count; i++)
            {
                writer.WriteElementString($"outputneuron_{i}", (output[i] as Neuron).Activation.ToString());
                for (int j = 0; j < (output[i] as Neuron).Synapses.Length; j++)
                {
                    writer.WriteElementString($"outputneuron_{i}_synapse_{j}_weight", (output[i] as Neuron).Synapses[j].Weight.ToString());
                }
            }



            writer.WriteEndElement();
            writer.Flush();
            writer.Close();
        }
    }
}
