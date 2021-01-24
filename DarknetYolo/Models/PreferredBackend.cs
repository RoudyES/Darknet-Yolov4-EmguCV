using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetYolo.Models
{
    public enum PreferredBackend
    {
        /// <summary>
        /// Default equals to InferenceEngine if OpenCV is built with Intel's Inference Engine library or Opencv otherwise.
        /// </summary> 
        Default = 0,
        /// <summary>
        /// Halide backend
        /// </summary>
        Halide = 1,
        /// <summary>
        /// Intel's Inference Engine library
        /// </summary>    Intel's Inference Engine library
        InferenceEngine = 2,
        /// <summary>
        /// OpenCV's implementation
        /// </summary>
        OpenCV = 3,
        /// <summary>
        /// Vulkan based backend
        /// </summary>
        VkCom = 4,
        /// <summary>
        /// Cuda backend
        /// </summary>
        Cuda = 5,
        /// <summary>
        /// Inference Engine NGraph
        /// </summary>
        InferenceEngineNgraph = 1000000,
        /// <summary>
        /// Inference Engine NN Builder 2019
        /// </summary>
        InferenceEngineNnBuilder2019 = 1000001
    }
}
