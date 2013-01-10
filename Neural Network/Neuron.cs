using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    class Neuron
    {
        // Attributes
        private static int bias = 1;
        private double[] input;
        private double[] weights;

        // Constructors
        public Neuron( ref double[] input )
        {
            SetInput(ref input);
            SortWeights();
        }

        // Setters
        public void SetInput(ref double[] input)
        {
            this.input = input;
        }

        public void SetWeights(ref double[] weights)
        {
            // instanciando classe para geração de números aleatórios
            Random random = new Random(DateTime.Now.Millisecond);

        }

        // Getters


        // Functional Methods
        /**
         * Sorteia o peso para cada entrada
         */
        private void SortWeights()
        { 
        
        }
    }
}
