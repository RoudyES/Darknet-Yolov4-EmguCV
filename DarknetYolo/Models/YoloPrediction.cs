using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetYolo.Models
{
    public class YoloPrediction
    {
        public Rectangle Rectangle { get; set; }

        public string Label { get; set; }

        public double Confidence { get; set; }
    }
}
