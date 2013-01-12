using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] inputs = { -1.0, -1.0 };
            double[] weights = { 0.4, 0.1, -0.1 };
            
            Neuron neuron = new Neuron( ref inputs );
            neuron.SetWeights(ref weights);
            neuron.Forward();

            Console.Write("Inputs: ");
            foreach (double value in neuron.GetInput())
                Console.Write("{0:f4}  ", value);
            Console.WriteLine();

            Console.Write("Weights:");
            foreach (double value in neuron.GetWeights())
                Console.Write("{0:f4}  ", value);
            Console.WriteLine();

            Console.Write("Output: {0:f4}", neuron.GetOutput());

            Console.ReadKey();
        }
    }
}
