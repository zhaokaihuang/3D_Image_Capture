//
//  main.cpp
//  3dImage
//
//  Created by DiaMond on 4/6/16.
//  Copyright © 2016 DiaMond. All rights reserved.
//

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

//include pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>

#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/filters/filter.h>

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/photo/photo.hpp>

#include "data.hpp"
#include "filtering.h"
#include "registration.h"
#include "surface.h"

using namespace cv;
using namespace std;
using namespace pcl;

const uint rowNumber = 480;
const uint colNumber = 640;

// implemented program contains:
// capture depth & RGB - in data.hpp
// compute point clouds - in data.hpp
// point cloud filtering - in filtering.h
// registration & merge - in registration.h
// surface reconstruction - in surface.h
int main(int argc, char** argv){
    
    cout << "opening device(s)" << endl;
    VideoCapture capture;
    capture.open(CAP_OPENNI);

    if( !capture.isOpened() ){
        cout << "Can not open capture object 1." << endl;
        return -1;
    }
    capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
    
    // Print some avalible device settings.
    cout << "\nDepth generator output mode:" << endl <<
    "FRAME_WIDTH      " << capture.get( CAP_PROP_FRAME_WIDTH ) << endl <<
    "FRAME_HEIGHT     " << capture.get( CAP_PROP_FRAME_HEIGHT ) << endl <<
    "FRAME_MAX_DEPTH  " << capture.get( CAP_PROP_OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
    "FPS              " << capture.get( CAP_PROP_FPS ) << endl <<
    "CV_CAP_PROP_OPENNI_BASELINE   " <<  capture.get( CV_CAP_PROP_OPENNI_BASELINE ) << endl <<
    "CV_CAP_PROP_OPENNI_FOCAL_LENGTH   " <<  capture.get( CV_CAP_PROP_OPENNI_FOCAL_LENGTH ) << endl <<
    "REGISTRATION     " << capture.get( CAP_PROP_OPENNI_REGISTRATION ) << endl;
    if( capture.get( CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
        "\nImage generator output mode:" << endl <<
        "FRAME_WIDTH   " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_WIDTH ) << endl <<
        "FRAME_HEIGHT  " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_HEIGHT ) << endl <<
        "FPS           " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator." << endl;
        return 0;
    }
    Mat depthMap, depthf, depthMap2;
    Mat bgrImage;
    Mat temp(Size(640, 480), CV_8UC1);

    int begin = 1;
    string str = "input/capture_";
    // capture depthMap & bgrImage
    for(;;){
        
        if( !capture.grab() ){
            cout << "Sensor can not grab images." << endl;
            return -1;
        }
        // depthmap CV_16UC1
        if( capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP ) )
        {
            //depth map grayscale
            Mat show, show2;
            double min, max;
            cv::minMaxLoc(depthMap, &min, &max);
            
            depthMap.convertTo( show, CV_8UC1, 255.0/max);
            imshow("depth map", show);
            
            // filter size = 11
            depthMap = bilateral(depthMap, 11);
        }
        
        // bgrImage CV_8UC3
        if( capture.retrieve( bgrImage, CAP_OPENNI_BGR_IMAGE ) ){
            imshow("bgrImage",bgrImage);
        }
        
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        pcl::PointCloud<pcl::PointXYZRGB> output;

        // source cloud
        cout<<"capture clouds "<<begin<<endl;

        cloud = computeColoredCloud(depthMap, bgrImage);
        
        pcl::io::savePCDFileBinary(str+to_string(begin)+".pcd", cloud);
        pcl::io::savePLYFileBinary(str+to_string(begin)+".ply", cloud);
        
        cv::imwrite(str+to_string(begin)+"_depth.jpg", depthMap);
        cv::imwrite(str+to_string(begin)+"_rgb.jpg", bgrImage);
        
        begin++;
        if( waitKey( 500 ) >=0)
            break;// ESC to exit
    }
    
    vector<pcl::PointCloud<pcl::PointXYZRGB>> models;
    // load captured clouds
    loadData(str, models);

    // initialize variables
    Eigen::Matrix4f globalTransform = Eigen::Matrix4f::Identity (), pairTransform;
    pcl::PointCloud<pcl::PointXYZRGB> model, cloud_current, cloud_ref, cloud_aligned, cloud_scene;

    cloud_ref = models[0];
    cloud_current = models[1];

    int mergesize = 0;
    pairTransform = pairwiseRegister(cloud_current, cloud_ref);
    transformPointCloud(cloud_current, cloud_aligned, pairTransform);
    cloud_scene = pcmerge(cloud_ref, cloud_aligned, mergesize);

    globalTransform = pairTransform;
    
    // fusing clouds
    // cloud n -> n-1, n-1 -> n-2, ..., 2 -> 1
    for(int i = 2; i<models.size(); i++){
        cout<<i<<endl;
        cloud_current = models[i];
        if(cloud_current.points.size()!=640*480)
            continue;

        cloud_ref = models[i-1];

        pairTransform = pairwiseRegister(cloud_current, cloud_ref);
        globalTransform = pairTransform * globalTransform;

        transformPointCloud(cloud_current, cloud_aligned, globalTransform);
        cloud_scene = pcmerge(cloud_scene, cloud_aligned, mergesize);

//        pcl::io::savePLYFile("output/cloud_scene"+to_string(i-1)+".ply", cloud_scene);
    }
    
    pcl::io::savePCDFile("output/cloud_scene.pcd", cloud_scene);
    pcl::io::savePLYFile("output/cloud_scene.ply", cloud_scene);

    pcl::VoxelGrid<pcl::PointXYZRGB> vox;
    vox.setInputCloud (cloud_scene.makeShared());
    vox.setLeafSize (0.02, 0.02, 0.02);
    vox.filter (cloud_scene);
//    pcl::io::savePLYFile("scene_voxel.ply", cloud_scene);


    PolygonMesh::Ptr mesh = gp3Reconstruct(cloud_scene.makeShared());
    pcl::io::savePLYFileBinary("scene_mesh.ply", *mesh);

}