//
//  filtering.h
//  3dImage
//
//  Created by DiaMond on 6/20/16.
//  Copyright © 2016 DiaMond. All rights reserved.
//

#ifndef filtering_h
#define filtering_h

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/filter.h>

#include <pcl/surface/mls.h>

using namespace std;
using namespace pcl;

//5.0f mm -> 0.005f meters; 0.01f meter
const float leafSize = 0.05f;
const float cut_lowDist = 0.0f;
const float cut_upDist = 3.5f;

// parameters dependent on size of input data
const float leafSize_text = 0.03f;
const float cut_lowDist_test = 1.8f;
const float cut_upDist_test = 4.2f;

// pass through filter
pcl::PointCloud<pcl::PointXYZRGB> pass_filter(pcl::PointCloud<pcl::PointXYZRGB> cloud){
    std::cerr << "PointCloud before pass through filter: " << cloud.width * cloud.height
    << " data points (" << pcl::getFieldsList (cloud) << ")."<<endl;

    pcl::PassThrough<PointXYZRGB> pass_through;
    pass_through.setInputCloud (cloud.makeShared());
    
    // points having values outside this interval for this field will be discarded.
    pass_through.setFilterLimits (cut_lowDist_test, cut_upDist_test);
    pass_through.setFilterFieldName ("z");
    pass_through.filter(cloud);
    
    std::cerr << "PointCloud after pass through filter: " << cloud.width * cloud.height
    << " data points (" << pcl::getFieldsList (cloud) << ")."<<endl;

    return cloud;
}


// Downsample
// Note that the downsampling step does not only speed up the registration,
// but can also improve the accuracy.
pcl::PointCloud<pcl::PointXYZRGB> voxel_filter(pcl::PointCloud<pcl::PointXYZRGB> cloud){
    if(cloud.points.size()<4500)
        return cloud;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
    
    std::cerr << "PointCloud before filtering: " << cloud.width * cloud.height
    << " data points (" << pcl::getFieldsList (cloud) << ")."<<endl;
    // Create the filtering object
    pcl::VoxelGrid<pcl::PointXYZRGB> vox;
    vox.setInputCloud (cloud.makeShared());
    vox.setLeafSize (leafSize_text, leafSize_text, leafSize_text);
    vox.filter (cloud_filtered);
    float size = leafSize;
    while(cloud_filtered.points.size()<4500){
        pcl::VoxelGrid<pcl::PointXYZRGB> vox2;
        size = size - 0.01;
        vox2.setInputCloud (cloud.makeShared());
        vox2.setLeafSize (size, size, size);
        vox2.filter (cloud_filtered);
        std::cerr << "PointCloud after filtering: " << cloud_filtered.width * cloud_filtered.height
        << " data points (" << pcl::getFieldsList (cloud_filtered) << ")."<<endl;

    }
    std::cerr << "PointCloud after filtering: " << cloud_filtered.width * cloud_filtered.height
    << " data points (" << pcl::getFieldsList (cloud_filtered) << ")."<<endl;
    
    return cloud_filtered;
}



// remove outlier
pcl::PointCloud<pcl::PointXYZRGB> outlier_filter(pcl::PointCloud<pcl::PointXYZRGB> cloud){
    std::cerr << "PointCloud before removing outliers: " << cloud.width * cloud.height
    << " data points (" << pcl::getFieldsList (cloud) << ")."<<endl;
    
    StatisticalOutlierRemoval<PointXYZRGB> outlier_filter;
    outlier_filter.setMeanK (50); // 50 points to estimate mean distance
    outlier_filter.setStddevMulThresh(1.0); // 1.0 standard deviation
    outlier_filter.setInputCloud(cloud.makeShared());
    outlier_filter.filter(cloud);
    
    std::cerr << "PointCloud after removing outliers: " << cloud.width * cloud.height
    << " data points (" << pcl::getFieldsList (cloud) << ")."<<endl;
    
    return cloud;
}

// MovingLeastSquares, resample data(unused in the project recently)
pcl::PointCloud<pcl::PointXYZRGB> mls_resample(pcl::PointCloud<pcl::PointXYZRGB> cloud){
    
    std::cerr << "PointCloud before MLS_resample: " << cloud.width * cloud.height
    << " data points (" << pcl::getFieldsList (cloud) << ")."<<endl;

    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB> mls_points, output;
    
    // Init object
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
    
    // Set parameters
    mls.setInputCloud (cloud.makeShared());
    mls.setSearchRadius (0.03);
    mls.setPolynomialFit (true);
    mls.setPolynomialOrder (2);
    mls.setUpsamplingMethod (pcl::MovingLeastSquares<PointXYZRGB, PointXYZRGB>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius (0.005);
    mls.setUpsamplingStepSize (0.003);

    // Reconstruct
    mls.process (mls_points);
    
    for(size_t i = 0; i<mls_points.size(); i++){
        if(!isnan(mls_points.points[i].x)&&!isnan(mls_points.points[i].y)&&!isnan(mls_points.points[i].z)){
            output.push_back(mls_points.points[i]);
        }
    }
    std::cerr << "PointCloud after MLS_resample: " << output.width * output.height
    << " data points (" << pcl::getFieldsList (output) << ")."<<endl;

    return output;
}

#endif /* filtering_h */
