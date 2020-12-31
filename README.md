# Darknet-Yolov4-EmguCV
This project reads Darknet models supported by the OpenCV library and displays inference result.

Works on both videos on images, to detect objects from images just uncomment the code block in the "Main" function and comment the video block.

# Models
You can download and test pre-trained models from [Here](https://hackmd.io/NFj2NrmqTcefjc2l94KjpQ).
This project was mainly tested on Yolov4-Tiny model due to the lack of GPU. Bigger models might require bigger image resolution which in turn will take more processing time.

# Cuda support
This implementation was done purely on the CPU. The downloaded [EmguCV](http://www.emgu.com/wiki/index.php/Main_Page) nuget package is only for windows runtime and on cpu. To try out the project on gpu, try downloading their [CUDA runtime](https://www.nuget.org/packages/Emgu.CV.runtime.windows.cuda/) and make sure to have all cuda drivers installed on your system. Also, make sure to change the preferred backend and target while loading Darknet model.
