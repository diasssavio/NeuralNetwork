using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    class MultiLayerPerceptronNetwork
    {
        // ------------------- 1.VARIABLES AND PROPERTIES -------------------
        public double[] Input { get; set; }
        public Neuron[][] HiddenLayer { get; set; }
        public Neuron[] OutLayer { get; set; }

        //  ------------------- 2.CONSTRUCTORS -------------------
        public MultiLayerPerceptronNetwork() { }

        public MultiLayerPerceptronNetwork(double[] input, int[] hiddenLayerAmount)
        {
            HiddenLayer = new Neuron[ hiddenLayerAmount.Length ][];
            for (int i = 0; i < HiddenLayer.Length; i++)
                HiddenLayer[i] = new Neuron[ hiddenLayerAmount[i] ];

            // Set input on the first hidden layer
            SetInputOnHiddenLayer(input);
        }

        //  ------------------- 3.GETTERS & SETTERS -------------------
        /**
         * Retorna o vetor com as saídas de cada neurônio
         */
        public double[] GetOutputs()
        {
            double[] output = new double[ OutLayer.Length ];
            for (int i = 0; i < OutLayer.Length; i++)
                output[i] = OutLayer[i].Output;

            return output;
        }

        /**
         * Define a entrada, input, nos neurônios da camada oculta, position
         */
        private void SetInputOnHiddenLayer(double[] input, int position = 0)
        {
            if (input != null)
            {
                // If network input isn't setted yet. First hidden layer case
                if (Input != null)
                    Input = input;

                // Put values "input" in the "position" layer
                for (int i = 0; i < HiddenLayer[position].Length; i++)
                    HiddenLayer[position][i] = new Neuron(input, 0.1);
            }
            else throw new Exception("input parameter invalid for function task!");
        }

        /**
         * Define a entrada, input, nos neurônios da camada de saída
         */
        private void SetInputOnOutLayer(double[] input)
        {
            if (input != null)
                for (int i = 0; i < OutLayer.Length; i++)
                    OutLayer[i] = new Neuron(input, 0.1);
            else throw new Exception("input parameter invalid for function task!");
        }

        //  ------------------- 4.FUNCTIONAL METHODS -------------------
        /**
         * Realiza o processo de propagação (forward) na rede
         */
        public void Forward()
        {
            // Do forward process in hidden layers
            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                // Vector to store the current neuron layer outs
                double[] tempOut = new double[ HiddenLayer[i].Length ];

                // Forward process in current hidden layer and store the outputs
                for (int j = 0; j < HiddenLayer[i].Length; j++)
                {
                    HiddenLayer[i][j].Forward();
                    tempOut[j] = HiddenLayer[i][j].Output;
                }

                // If the hidden layers is over, then set inputs on output layer
                if (i + 1 == HiddenLayer.Length)
                    SetInputOnOutLayer(tempOut);
                else
                    SetInputOnHiddenLayer(tempOut, i + 1);
            }

            // Do forward process in the output layer
            for (int i = 0; i < OutLayer.Length; i++)
                OutLayer[i].Forward();
        }
    }
}
