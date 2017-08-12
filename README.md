# 3D_Image_Capture
A project enables the generation of 3D image using **OpenCV** and **PCL**, based on **Microsoft Kinect**.

Implemented program contains:
* capture depth & RGB - in [data.hpp](./src/data.hpp)
* compute point clouds - in [data.hpp](./src/data.hpp)
* point cloud filtering - in [filtering.h](./src/filtering.h)
* [ICP](https://en.wikipedia.org/wiki/Iterative_closest_point) registration & merge - in [registration.h](./src/registration.h)
* surface reconstruction(Poission, Marching Cubes Algorithm) - in [surface.h](./src/surface.h)
