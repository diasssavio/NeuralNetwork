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
            BackwardMLP();

            Console.ReadKey();
        }

        #region Neuron Tests
        static void ForwardNeuron()
        {
            double[] inputs = { -1.0, -1.0 };
            // { bias, entradas... }
            double[] weights = { -0.1, 0.4, 0.1 };

            Neuron neuron = new Neuron(inputs, 0.1);
            neuron.Weights = weights;
            neuron.Forward();

            Console.WriteLine("bias: {0}", Neuron.Bias);

            Console.Write("Inputs: ");
            foreach (double value in neuron.Input)
                Console.Write("{0:f4}  ", value);
            Console.WriteLine();

            Console.Write("Weights:");
            foreach (double value in neuron.Weights)
                Console.Write("{0:f4}  ", value);
            Console.WriteLine();

            Console.Write("Output: {0:f4}", neuron.Output);
        }

        static void BackwardNeuron()
        {
            Neuron neuron = new Neuron(0.1);
            neuron.Weights = new double[] { -1.0, 1.0, 0.5 };

            for (int i = 1; i <= 10; i++)
            {
                neuron.Backward(new double[] { -1.0, -1.0 }, -1.0);
                Console.WriteLine("{0}: -1.0, -1.0 = {1:f6}", i, neuron.Output);

                neuron.Backward(new double[] { 1.0, -1.0 }, -1.0);
                Console.WriteLine("{0}: 1.0, -1.0 = {1:f6}", i, neuron.Output);

                neuron.Backward(new double[] { -1.0, 1.0 }, -1.0);
                Console.WriteLine("{0}: -1.0, 1.0 = {1:f6}", i, neuron.Output);

                neuron.Backward(new double[] { 1.0, 1.0 }, 1.0);
                Console.WriteLine("{0}: 1.0, 1.0 = {1:f6}", i, neuron.Output);

                Console.WriteLine();
            }
        }
        #endregion

        #region MultiLayerPerceptronNetwork Tests
        static void ForwardMLP()
        {
            MultiLayerPerceptronNetwork network = new MultiLayerPerceptronNetwork( new double[] { -1, -1 }, 2, 1, 0.5 );
            network.Forward();

            Console.WriteLine("MLP output: ");
            foreach (double output in network.GetOutputs())
                Console.Write("{0}\t", output);
        }

        static void BackwardMLP()
        {
            MultiLayerPerceptronNetwork network = new MultiLayerPerceptronNetwork(2, 1, 0.5);
            
            Console.WriteLine("Training network...");
            for (int i = 0; i < 100; i++)
            {
                network.Backward(new double[] { -1.0, -1.0 }, new double[]{ -1.0 });
                Console.WriteLine("{0}: -1.0, -1.0 = {1:f6}", i, network.GetOutputs()[0]);

                network.Backward(new double[] { -1.0, 1.0 }, new double[]{ 1.0 });
                Console.WriteLine("{0}: -1.0, 1.0 = {1:f6}", i, network.GetOutputs()[0]);

                network.Backward(new double[] { 1.0, -1.0 }, new double[]{ 1.0 });
                Console.WriteLine("{0}: 1.0, -1.0 = {1:f6}", i, network.GetOutputs()[0]);

                network.Backward(new double[] { 1.0, 1.0 }, new double[]{ -1.0 });
                Console.WriteLine("{0}: 1.0, 1.0 = {1:f6}", i, network.GetOutputs()[0]);

                Console.WriteLine();
            }
        }
        #endregion
    }
}
