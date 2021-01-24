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
using DarknetYolo.Models;
using System.IO;

namespace DarknetYolo
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
            Enum.TryParse(backend.ToString(), out Emgu.CV.Dnn.Backend b);
            Enum.TryParse(target.ToString(), out Emgu.CV.Dnn.Target t);
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
            if (resizedWidth % 32 is int rest)
            {
                if (resizedWidth < 32)
                    resizedWidth = 32;
                if (rest < 16)
                    resizedWidth = (int)(32 * Math.Floor(resizedWidth / 32f));
                else
                    resizedWidth = (int)(32 * Math.Ceiling(resizedWidth / 32f));
            }

            if (resizedHeight % 32 is int rest2)
            {
                if (resizedHeight < 32)
                    resizedHeight = 32;
                if (rest2 < 16)
                    resizedHeight = (int)(32 * Math.Floor(resizedHeight / 32f));
                else
                    resizedHeight = (int)(32 * Math.Ceiling(resizedHeight / 32f));
            }

            Mat t = new Mat();
            int width = inputImage.Width;
            int height = inputImage.Height;
            VectorOfMat layerOutputs = new VectorOfMat();
            string[] outNames = Network.UnconnectedOutLayersNames;
            var blob = DnnInvoke.BlobFromImage(inputImage.ToImage<Bgr, byte>(), 1 / 255f, new System.Drawing.Size(resizedWidth, resizedHeight), swapRB: true, crop: false);
            Network.SetInput(blob);
            Network.Forward(layerOutputs, outNames);

            List<Rectangle> boxes = new List<Rectangle>();
            List<float> confidences = new List<float>();
            List<int> classIDs = new List<int>();
            for (int k = 0; k < layerOutputs.Size; k++)
            {
                float[,] lo = (float[,])layerOutputs[k].GetData();
                int len = lo.GetLength(0);
                for (int i = 0; i < len; i++)
                {
                    if (lo[i, 4] < ConfidenceThreshold)
                        continue;
                    float max = 0;
                    int idx = 0;

                    int len2 = lo.GetLength(1);
                    for (int j = 5; j < len2; j++)
                        if (lo[i, j] > max)
                        {
                            max = lo[i, j];
                            idx = j - 5;
                        }

                    if (max > ConfidenceThreshold)
                    {
                        lo[i, 0] *= width;
                        lo[i, 1] *= height;
                        lo[i, 2] *= width;
                        lo[i, 3] *= height;

                        int x = (int)(lo[i, 0] - (lo[i, 2] / 2));
                        int y = (int)(lo[i, 1] - (lo[i, 3] / 2));

                        var rect = new Rectangle(x, y, (int)lo[i, 2], (int)lo[i, 3]);

                        rect.X = rect.X < 0 ? 0 : rect.X;
                        rect.X = rect.X > width ? width - 1 : rect.X;
                        rect.Y = rect.Y < 0 ? 0 : rect.Y;
                        rect.Y = rect.Y > height ? height - 1 : rect.Y;
                        rect.Width = rect.X + rect.Width > width ? width - rect.X - 1 : rect.Width;
                        rect.Height = rect.Y + rect.Height > height ? height - rect.Y - 1 : rect.Height;

                        boxes.Add(rect);
                        confidences.Add(max);
                        classIDs.Add(idx);
                    }
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
