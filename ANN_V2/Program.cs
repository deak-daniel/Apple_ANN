using System.Runtime.Serialization.Formatters.Binary;
using static System.Net.Mime.MediaTypeNames;

namespace ANN_V2
{
    internal class Program
    {
        public static Random rnd = new Random();
        public static NeuralNetwork network;
        static void Main(string[] args)
        {
            // Separating dataset into Train and test subsets.
            #region Data preparation
            List<string> input = File.ReadAllLines("milknew.csv").Skip(1).Select(x => x).ToList();
            List<string> train = new List<string>();
            List<string> test = new List<string>();
            for (int i = 0; i < (int)(input.Count * 0.8); i++) 
            {
                train.Add(input[i]);
            }
            for (int i = (int)(input.Count * 0.8) + 1; i < input.Count - 1; i++)
            {   
                test.Add(input[i]);
            }
            #endregion

            network = new NeuralNetwork(input[1].Split(',').Length - 1);
            Train(train);

            Console.WriteLine($"Network trained!");

            // Writing the trained neural network to a file
            if (network.IsTrained)
            {
                using (FileStream fs1 = new FileStream("data.dat", FileMode.Create))
                {
                    BinaryFormatter formatter = new BinaryFormatter();
                    formatter.Serialize(fs1, network);
                }
            }

            Test(test);

        }
        /// <summary>
        /// Shuffling the data in order to avoid local minimums while training.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public static List<string> ShuffleData (List<string> data)
        {
            string[] strings = new string[data.Count];
            for (int j = 0; j < strings.Length; j++)
            {
                strings[j] = "";
            }

            int i = 0;
            while(i < data.Count)
            {
                int index = rnd.Next(0, data.Count);
                if (strings[index] == "")
                {
                    strings[index] = data[i];
                    i++;
                }
            }

            return strings.ToList();
        }
        /// <summary>
        /// Testing loop for the network.
        /// </summary>
        /// <param name="test">Test dataset</param>
        public static void Test(List<string> test)
        {
            for (int i = 0; i < test.Count - 1; i++)
            {
                List<string> helper = test[i].Split(",").ToList();
                network.Initialize(helper);

                network.Feedforward();

                Console.WriteLine($"Error: {network.error}, Actual: {network.Actual}, Predicted: {network.Chosen.Id}");
            }
        }
        /// <summary>
        /// Training loop for the network.
        /// </summary>
        /// <param name="train">Training dataset</param>
        /// <param name="epochs">Iterations.</param>
        /// <param name="errorThreshold">the threshold which has to be achieved by the network.</param>
        public static void Train(List<string> train, int epochs = 50000, double errorThreshold = 0.0001)
        {
            double sum = 0;
            double sum2 = 0;
            int epochCounter = 0;
            while (epochCounter < epochs /*&& (network.error > errorThreshold)*/)
            {
                for (int i = 0; i < train.Count; i++)
                {
                    List<string> helper = train[i].Split(",").ToList();
                    //List<string> assd = new List<string>();
                    //for (int z = 1; z < helper.Count; z++)
                    //{
                    //    assd.Add(helper[z]);
                    //}
                    network.Initialize(helper);

                    network.Feedforward();
                    network.Error();
                    sum += network.Error();

                    network.Backpropagate();
                }
                train = ShuffleData(train);
                Console.WriteLine($"Average Error: {sum / train.Count}");
                //Console.WriteLine($"Actual: {network.Actual}, Predicted: {network.Chosen.Id}");
                sum = 0;
                sum2 = 0;
                epochCounter++;
            }
            if (network.error < errorThreshold)
            {
                network.IsTrained = true;
            }

        }
    }
}
