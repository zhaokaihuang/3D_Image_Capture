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

const int gaussianVal = 6;
const int medianVal = 10;
const int bilateralVal = 10;
const int MAX_KERNEL_LENGTH = 31;

int main(int argc, char** argv){
    
    string str = "experiment/";
    
    Mat depthMap, depthMap2, filtered, final;
    Mat bgrImage;
    
    double error = 1000000;
    depthMap = imread(str+"depth_5.png", CV_LOAD_IMAGE_ANYDEPTH);
    bgrImage = imread(str+"rgb_5.jpg");
    
    cout<<type2str(depthMap.type())<<endl;
    //cvtColor(depthMap, depthMap, COLOR_BGR2GRAY);
    
    cout<<depthMap.type()<<endl;
    // depthmap CV_16UC1
    double min, max;
    cv::minMaxLoc(depthMap, &min, &max);
    
    // grayscale img
    
    depthMap.convertTo( depthMap2, CV_8UC1, 255.0/max);
    
//    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 ){
////        GaussianBlur( depthMap2, filtered, Size( i, i ), 0, 0);
////                    medianBlur(depthMap2, filtered, i );
//                    bilateralFilter(depthMap2, filtered, i, i*2, i/2);
//        double mserror = mse(depthMap2, filtered);
//        cout<<i<<": "<<mserror<<endl;
//        cout<<i<<": "<<psnr(mserror)<<endl;
//        cout<<endl;
//    }
    
    bilateralFilter(depthMap2, final, 13, 13*2, 13/2);
//    medianBlur(depthMap2, final, 11);
//    GaussianBlur( depthMap2, final, Size( 11, 11 ), 0, 0);
    
//    double min1;
//    double max1;
//    cv::minMaxIdx(final, &min1, &max1);
//    cv::Mat adjMap;
//    cv::convertScaleAbs(final, adjMap, 255 / max);
//    imwrite("bilateral11.png", final);
//    imwrite("ref_depth.png",depthMap2);
    for(;;){
//        cout<<error<<endl;
        imshow("depth map", depthMap2);

        imshow("filtered depth map", final);

        //depthMap2.convertTo(depthMap2, CV_16UC1, max/255.0);
    
        imshow("bgr image", bgrImage);
    
        if( waitKey( 1000 ) >=0)
            break;// ESC to exit
    }
    
    final.convertTo(final, CV_16UC1, max/255.0);

//    for(int i = 0; i<final.rows; i++){
//        for(int j = 0; j<final.cols; j++){
//            cout<<final.at<ushort>(i,j)<<endl;
//        }
//    }

    
    pcl::PointCloud<pcl::PointXYZRGB> cloud, filteredCloud;

    cloud = computeColoredCloud(depthMap, bgrImage);
    filteredCloud = computeColoredCloud(final, bgrImage);

    pcl::io::savePLYFileBinary("original3.ply", cloud);
    pcl::io::savePLYFileBinary("filteredCloud3.ply", filteredCloud);
    pcl::io::savePCDFileBinary("filteredCloud3.pcd", filteredCloud);

}




//int main(){
//    pcl::PointCloud<pcl::PointXYZRGB> alignedCloud;
//    pcl::PointCloud<pcl::PointXYZRGB> targetCloud, fullTarget;
//    pcl::PointCloud<pcl::PointXYZRGB> sourceCloud, fullSource;
//    PointCloud<PointXYZRGB>::Ptr my_cloud (new PointCloud<PointXYZRGB>);
//    //targetCloud.is_dense = true;
//    
//    if( pcl::io::loadPCDFile<pcl::PointXYZRGB> ("results/src.pcd", fullSource) == -1){
//        cout<<"cannot read"<<endl;
//    }
//    if( pcl::io::loadPCDFile<pcl::PointXYZRGB> ("results/tgt.pcd", fullTarget) == -1){
//        cout<<"cannot read"<<endl;
//    }
//    sourceCloud = pass_filter(fullSource);
//    sourceCloud = voxel_filter(sourceCloud);
//    
//    targetCloud = pass_filter(fullTarget);
//    targetCloud = voxel_filter(targetCloud);
//    
//    cout<<"Filtering finished."<<endl;
//    Eigen::Matrix4f globalTransform = Eigen::Matrix4f::Identity();
//    Eigen::Matrix4f initTransform, finalTransform;
//    
//    pcl::PointCloud<pcl::Normal>::Ptr sourceNormals = computeNormals(sourceCloud, fullSource);
//    pcl::PointCloud<pcl::Normal>::Ptr targetNormals = computeNormals(targetCloud, fullTarget);
//    
//    PointCloud<FPFHSignature33>::Ptr sourceFeatures = computeFeatures(sourceCloud, sourceNormals);
//    PointCloud<FPFHSignature33>::Ptr targetFeatures = computeFeatures(targetCloud, targetNormals);
//    
//    initTransform = sac_ia_transMatrix(sourceCloud, targetCloud, sourceFeatures, targetFeatures);
//    
//    //initTransform = sac_ia_transMatrix(sourceCloud, targetCloud);
//    globalTransform = globalTransform * initTransform;
//    transformPointCloud(sourceCloud,alignedCloud,globalTransform);
//    pcl::io::savePLYFileBinary("results/sac_ia_alignment.ply", alignedCloud);
//    
//    finalTransform = icp_transMatrix(alignedCloud, targetCloud);
//    globalTransform = globalTransform * finalTransform;
//    transformPointCloud(fullSource,alignedCloud,globalTransform);
//    pcl::io::savePLYFileBinary("results/aligned_cloud.ply", alignedCloud);
//    alignedCloud += fullTarget;
//    pcl::io::savePLYFileBinary("results/merged_cloud.ply", alignedCloud);
//    
//}


