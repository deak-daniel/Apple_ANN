using System.Runtime.Serialization.Formatters.Binary;

namespace ANN_V2
{
    internal class Program
    {
        public static Random rnd = new Random();    
        static void Main(string[] args)
        {
            List<string> input = File.ReadAllLines("apple_quality.csv").Skip(1).Select(x => x).ToList();
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
            NeuralNetwork network = new NeuralNetwork(input[0].Split(',').Length);
            int epochs = 50000;
            double sum = 0;
            double sum2 = 0;
            int epochCounter = 0;
            while (epochCounter < epochs && (network.error > 0.0001))
            {

                for (int i = 0; i < train.Count; i++)
                {
                    List<string> helper = train[i].Split(",").ToList();

                    network.Initialize(helper);

                    network.Feedforward();
                    network.error = network.Error();
                    network.errorDerivative = network.ErrorDerivative();
                    sum += network.Error();
                    sum2 += network.ErrorDerivative();
                    network.Backpropagate();


                }
                network.error = sum / train.Count;
                network.errorDerivative = sum2 / train.Count;
                train = ShuffleData(train);
                Console.WriteLine($"Average Error: {network.error}");
                sum = 0;
                sum2 = 0;
                epochCounter++;
            }

            Console.WriteLine($"Network trained!");


            for (int i = 0; i < test.Count - 1; i++)
            {
                List<string> helper = test[i].Split(",").ToList();
                network.Initialize(helper);

                network.Feedforward();

                Console.WriteLine($"Error: {network.error}, Actual: {network.Actual}, Predicted: {network.Predicted}");
            }



            using (FileStream fs1 = new FileStream("data.dat", FileMode.Create))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(fs1, network);
            }

            //FileStream fs = new FileStream("data.dat", FileMode.Open);
            //BinaryFormatter formatter2 = new BinaryFormatter();
            //NeuralNetwork myObject = (NeuralNetwork)formatter2.Deserialize(fs);
        }
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
    }
}
