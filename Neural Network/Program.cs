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
            LearningNeuron();

            Console.ReadKey();
        }

        static void ForwardNeuron()
        {
            double[] inputs = { -1.0, -1.0 };
            // { bias, entradas... }
            double[] weights = { -0.1, 0.4, 0.1 };

            Neuron neuron = new Neuron(inputs);
            neuron.SetWeights(weights);
            neuron.Forward();

            Console.WriteLine("bias: {0}", Neuron.GetBias());

            Console.Write("Inputs: ");
            foreach (double value in neuron.GetInputs())
                Console.Write("{0:f4}  ", value);
            Console.WriteLine();

            Console.Write("Weights:");
            foreach (double value in neuron.GetWeights())
                Console.Write("{0:f4}  ", value);
            Console.WriteLine();

            Console.Write("Output: {0:f4}", neuron.GetOutput());
        }

        static void LearningNeuron()
        {
            Neuron neuron = new Neuron();
            neuron.SetWeights(new double[] { -1.0, 1.0, 0.5 });

            for (int i = 1; i <= 10; i++)
            {
                neuron.Backward(new double[] { -1.0, -1.0 }, -1.0);
                Console.WriteLine("{0}: -1.0, -1.0 = {1:f6}", i, neuron.GetOutput());
                neuron.Backward(new double[] { 1.0, -1.0 }, -1.0);
                Console.WriteLine("{0}: 1.0, -1.0 = {1:f6}", i, neuron.GetOutput());
                neuron.Backward(new double[] { -1.0, 1.0 }, -1.0);
                Console.WriteLine("{0}: -1.0, 1.0 = {1:f6}", i, neuron.GetOutput());
                neuron.Backward(new double[] { 1.0, 1.0 }, 1.0);
                Console.WriteLine("{0}: 1.0, 1.0 = {1:f6}", i, neuron.GetOutput());
                Console.WriteLine();
            }

            //double[] inputs = { 1.0, 1.0 };
            //Console.WriteLine("  ");
        }
    }
}
