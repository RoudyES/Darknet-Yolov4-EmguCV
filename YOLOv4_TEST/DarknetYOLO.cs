using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.Dnn;
using Emgu.CV;
using Emgu.CV.Util;
using System.Drawing;
using Emgu.CV.Structure;
using DarknetYOLOv4.Models;
using System.IO;

namespace DarknetYOLOv4
{
    public class DarknetYOLO
    {
        /// <summary>
        /// Model to load
        /// </summary>
        public Net Network { get; set; }

        /// <summary>
        /// Prediction confidence threshold
        /// </summary>
        public float ConfidenceThreshold { get; set; }

        /// <summary>
        /// Non-Max Suppression Threshold
        /// </summary>
        public float NMSThreshold { get; set; }

        private string[] _labels;

        /// <summary>
        /// Initialize Darknet network.
        /// </summary>
        /// <param name="labelsPath">Path to the labels file.</param>
        /// <param name="weightsPath">Path to the weights file.</param>
        /// <param name="configPath">Path to the config file.</param>
        /// <param name="backend">Preferred computation implementation.</param>
        /// <param name="target">Preferred computation target.</param>
        public DarknetYOLO(string labelsPath, string weightsPath, string configPath, PreferredBackend backend = PreferredBackend.OpenCV, PreferredTarget target = PreferredTarget.Cpu)
        {
            Emgu.CV.Dnn.Backend b;
            Emgu.CV.Dnn.Target t;
            Enum.TryParse(backend.ToString(), out b);
            Enum.TryParse(target.ToString(), out t);
            Network = DnnInvoke.ReadNetFromDarknet(configPath, weightsPath);
            Network.SetPreferableBackend(b);
            Network.SetPreferableTarget(t);
            _labels = File.ReadAllLines(labelsPath);
        }

        /// <summary>
        /// Detect objects from image.
        /// </summary>
        /// <param name="inputImage">The network's input image.</param>
        /// <param name="resizedWidth">(Optional) Resize image width before feeding it to the network (smaller results in faster predictions but may hurt accuracy).</param>
        /// <param name="resizedHeight">(Optional) Resize image height before feeding it to the network (smaller results in faster predictions but may hurt accuracy)</param>
        /// <returns>List of all detected objects.</returns>
        public List<YoloPrediction> Predict(Bitmap inputImage, int resizedWidth = 512, int resizedHeight = 512)
        {
            Mat layerOutputs = new Mat();

            Image<Bgr, byte> tmp = inputImage.ToImage<Bgr, byte>();
            Mat input = tmp.Mat.Clone();
            tmp.Dispose();
            var blob = DnnInvoke.BlobFromImage(input, 1 / 255.0, new System.Drawing.Size(resizedWidth, resizedHeight), swapRB: true, crop: false);
            Network.SetInput(blob);
            Network.Forward(layerOutputs);

            List<Rectangle> boxes = new List<Rectangle>();
            List<float> confidences = new List<float>();
            List<int> classIDs = new List<int>();

            float[,] lo = (float[,])layerOutputs.GetData();
            for (int i = 0; i < lo.GetLength(0); i++)
            {
                List<float> scores = new List<float>();
                List<float> bb = new List<float>();
                float confidence = 0;


                for (int j = 0; j < lo.GetLength(1); j++)
                {
                    if (j > 4)
                        scores.Add(lo[i, j]);
                    else
                        bb.Add(lo[i, j]);


                }

                int indexMax = !scores.Any() ? -1 :
                                scores
                                .Select((value, index) => new { Value = value, Index = index })
                                .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                                .Index;
                if (indexMax != -1)
                    confidence = scores[indexMax];

                if (confidence > ConfidenceThreshold)
                {
                    bb[0] *= input.Width;
                    bb[1] *= input.Height;
                    bb[2] *= input.Width;
                    bb[3] *= input.Height;

                    int x = (int)(bb[0] - (bb[2] / 2));
                    int y = (int)(bb[1] - (bb[3] / 2));

                    boxes.Add(new Rectangle(x, y, (int)bb[2], (int)bb[3]));
                    confidences.Add(confidence);
                    classIDs.Add(indexMax);
                }
            }

            int[] bIndexes = DnnInvoke.NMSBoxes(boxes.ToArray(), confidences.ToArray(), ConfidenceThreshold, NMSThreshold);

            List<YoloPrediction> filteredBoxes = new List<YoloPrediction>();
            if (bIndexes.Length > 0)
            {
                foreach (var idx in bIndexes)
                {
                    filteredBoxes.Add(new YoloPrediction()
                    {
                        Rectangle = boxes[idx],
                        Confidence = Math.Round(confidences[idx], 4),
                        Label = _labels[classIDs[idx]]
                    });
                }
            }
            return filteredBoxes;
        }
    }
}
