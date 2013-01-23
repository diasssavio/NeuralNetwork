using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    class PerceptronNetwork
    {
        // TODO - calcular erro dentro da rede e apenas "setar" na classe neurônio
        // ------------------- 1.VARIABLES AND PROPERTIES -------------------
        public double[] Input { get; set; }
        public Neuron[] Neurons { get; set; }

        //  ------------------- 2.CONSTRUCTORS -------------------
        public PerceptronNetwork(){ }

        public PerceptronNetwork(double[] input, int neuronsAmount)
        {
            Input = input;
            Neurons = new Neuron[neuronsAmount];
            for (int i = 0; i < neuronsAmount; i++)
                Neurons[i] = new Neuron(Input, 0.1);
        }

        //  ------------------- 3.GETTERS ------------------- 
        public double[] GetOutputs()
        { 
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < outputs.Length; i++)
                outputs[i] = Neurons[i].Output;

            return outputs;
        }

        //  ------------------- 4.FUNCTIONAL METHODS -------------------
        public void Forward()
        {
            foreach (Neuron neuron in Neurons)
                neuron.Forward();
        }


    }
}
