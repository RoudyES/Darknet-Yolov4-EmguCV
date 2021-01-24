using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetYolo.Models
{
    public enum PreferredTarget
    {
        /// <summary>
        /// CPU
        /// </summary>
        Cpu = 0,
        /// <summary>
        /// OpenCL
        /// </summary>
        OpenCL = 1,
        /// <summary>
        /// Will fall back to OPENCL if the hardware does not support FP16
        /// </summary>
        OpenCLFp16 = 2,
        /// <summary>
        /// Myriad
        /// </summary>
        Myriad = 3,
        /// <summary>
        /// Vulkan
        /// </summary>
        Vulkan = 4,
        /// <summary>
        /// FPGA device with CPU fallbacks using Inference Engine's Heterogeneous plugin.
        /// </summary>
        FPGA = 5,
        /// <summary>
        /// Cuda
        /// </summary>
        Cuda = 6,
        /// <summary>
        /// Cuda Fp16
        /// </summary>
        CudaFp16 = 7
    }
}
