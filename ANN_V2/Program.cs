namespace ANN_V2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            List<string> input = File.ReadAllLines("WineQT.csv").Skip(1).Select(x => x).ToList();
            List<string> train = new List<string>();
            List<string> test = new List<string>();
            for (int i = 0; i < (int)(input.Count * 0.8); i++) 
            {
                train.Add(input[i]);
            }
            for (int i = (int)(input.Count * 0.8) + 1; i < input.Count; i++)
            {
                test.Add(input[i]);
            }
            NeuralNetwork network = new NeuralNetwork(input[0].Split(',').Length);
            int epochs = 500;
            double sum = 0;
            for (int h = 0; h < epochs; h++)
            {

                for (int i = 0; i < train.Count; i++)
                {
                    List<string> helper = train[i].Split(",").ToList();
                    network.Initialize(helper);

                    network.Feedforward();

                    sum += network.Error();

                    network.Backpropagate();
                }
                Console.WriteLine($"Average Error: {sum}");
                sum = 0;
            }


            //for (int i = 0; i < train.Count; i++)
            //{
            //    List<string> helper = train[i].Split(",").ToList();
            //    network.Initialize(helper);

            //    network.Feedforward();

            //    Console.WriteLine( $"Error: {network.Error()}");

            //    network.Backpropagate();
            //}
            Console.WriteLine($"Network trained!");


            for (int i = 0; i < test.Count - 1; i++)
            {
                List<string> helper = test[i].Split(",").ToList();
                network.Initialize(helper);

                network.Feedforward();

                Console.WriteLine($"Error: {network.Error()}");
            }
        }
    }
}
