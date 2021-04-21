# Darknet-Yolov4-EmguCV
This project reads Darknet models supported by the OpenCV library and displays inference result.

Works on both videos on images, to detect objects from images just uncomment the code block in the "Main" function and comment the video block.

# Models
You can download and test pre-trained models from [Here](https://hackmd.io/NFj2NrmqTcefjc2l94KjpQ).
This project was mainly tested on Yolov4-Tiny model due to the lack of GPU. Bigger models might require bigger image resolution which in turn will take more processing time.

# Cuda support
This implementation supports CUDA. For smaller repo size, EmguCV's cuda libraries weren't included with the repo. In order to properly run, [Download EmguCV with CUDA](https://sourceforge.net/projects/emgucv/files/emgucv/4.4.0/libemgucv-windesktop_x64-cuda-4.4.0.4099.zip.selfextract.exe/download) and extract it. After extracting, copy the "libs" folder included in the extracted folder and paste it inside References/EmguCV (replace the existing libs folder).

# Demo
This is a demo on the model running on a laptop with: CPU - Intel Core i7 7700HQ, GPU - NVIDIA GeForce GTX 1050Ti.
![alt text](demo.gif?raw=true)

Performance on a 416x416 resolution:
* Frame processing time + Post Processing: <b>56-59ms</b> on average.
* Frames Per Second: <b>17-18 FPS</b>.
