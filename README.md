# Gaussian-Blur

In this project, we will be exploring how to implement a Gaussian Blurring program that will inherently alter each pixel of an image in order to take a new pixel value that will be affected by a pixel's surrounding neighbors. 
![Image Alt Text](https://github.com/MrGrinchFx/Gaussian-Blur/raw/main/building.png) ![Image Alt Text](https://github.com/MrGrinchFx/Gaussian-Blur/raw/main/blurredBuilding.png)

A naive approach to “blurring” an image would be to visit each pixel take the greyscale value of the surrounding 8 pixels and average it with the current pixel. However, this strategy leads to a muddier and less appealing result and is often not used in the real world. Instead, a different approach, the Gaussian Blur, is more common in real-world applications such as Blender or photo editors.

# Usage:
1. Make
2. For CPU Implementation: ./gaussian_blur_serial <path_to_pgm_file>
3. For CUDA Implemenation: ./gaussian_blur_cuda <path_to_pgm_file> (Please be sure that your system is CUDA capable)

