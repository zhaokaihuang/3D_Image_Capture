//
//  registration.h
//  3dImage
//
//  Created by DiaMond on 6/21/16.
//  Copyright © 2016 DiaMond. All rights reserved.
//

#ifndef registration_h
#define registration_h

#include <limits>
#include <fstream>
#include <vector>

#include <Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/crop_box.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>

#include <ctime>

#include "filtering.h"

using namespace pcl;
using namespace std;

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
typedef Eigen::Matrix4f Matrix;
typedef pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointXYZRGB> Keypoint;
typedef pcl::search::KdTree<pcl::PointXYZRGB> KdTree;
typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> NormalEstimation;
typedef pcl::PointCloud<pcl::FPFHSignature33> Features;
typedef pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> FeatureEstimation;
typedef pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> CorrespondenceEstimation;
typedef pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> CorrespondenceRejector;
typedef pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB> TransformationEstimation;
typedef pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> IterativeClosestPoint;


// feature_radius > normals_radius, always 0.1, 0.3
//radius of 0.05m for normal estimation & 0.25m for FPFH support size are good for Kinect point clouds
const float NORMALS_RADIUS = 0.05f;
const float FEATURES_RADIUS = 0.25f;

// alignment parameters, needs tune
const int MAX_SACIA_ITE = 500;
const float MIN_SAMPLE_DISTANCE = 0.05f; //0.4f
const float SAC_MAX_CORRESPONDENCE_DIST = 0.001f; // 0.01f*0.01f

// alignment parameters, needs tune
//const int MAX_SACIA_ITE = 500;
//const float MIN_SAMPLE_DISTANCE = 0.5f; //0.4f
//const float SAC_MAX_CORRESPONDENCE_DIST = 0.05f; // 0.01f*0.01f

const int MAX_ICP_ITE = 500;
const float ICP_MAX_CORRESPONDENCE_DIST = 0.01f;
const float ICP_TRANSFORMATION_EPSILON = 1e-8;
const float ICP_EUCLIDEAN_FITNESS_EPSILON = 1e-6;

// compute surface normals for features
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZRGB> cloud){
    
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    
    // Normals
    cout<<"Computing normals..."<<endl;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(cloud.makeShared());
    
    pcl::NormalEstimation<PointXYZRGB, Normal> normEst;
    
    // Set parameters
    normEst.setInputCloud (cloud.makeShared());
    normEst.setSearchMethod (kdTree);
    normEst.setRadiusSearch (NORMALS_RADIUS);
    normEst.compute (*normals);
    
    cout<<"normal size"<<(*normals).size()<<endl;
    return normals;
}

double computeCloudMSE(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr target, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr source, double max_range){
    
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(target);
    
    double fitness_score = 0.0;
    
    std::vector<int> nn_indices (1);
    std::vector<float> nn_dists (1);
    
    int nr = 0;
    for (size_t i = 0; i < source->points.size (); ++i){
        
        if(!pcl_isfinite((*source)[i].x)){
            continue;
        }
        
        // Find its nearest neighbor
        tree->nearestKSearch (source->points[i], 1, nn_indices, nn_dists); // squared_distance
        if (nn_dists[0] <= max_range*max_range){
            fitness_score += nn_dists[0];
//            cout<<fitness_score<<endl;
            nr++;
        }
        
    }
    
    return (fitness_score / nr);

}

// add searchSurface
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZRGB> cloud, pcl::PointCloud<pcl::PointXYZRGB> fullCloud){
    
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    
    // Normals
    cout<<"Computing normals..."<<endl;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(cloud.makeShared());

    pcl::NormalEstimation<PointXYZRGB, Normal> normEst;
    
    // Set parameters
    normEst.setInputCloud (cloud.makeShared());
    // Pass the original data (before downsampling) as the search surface
    normEst.setSearchSurface (fullCloud.makeShared());
    normEst.setSearchMethod (kdTree);
    normEst.setRadiusSearch (NORMALS_RADIUS);
    normEst.compute (*normals);
    
    return normals;
}


// Fast point feature histogram - feature descriptor for SAC-IA algo.
pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFeatures(pcl::PointCloud<pcl::PointXYZRGB> cloud, pcl::PointCloud<pcl::Normal>::Ptr normals){
    
    // Features
    cout<<"Computing features..."<<endl;
    PointCloud<FPFHSignature33>::Ptr features (new PointCloud<FPFHSignature33>);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree (new pcl::search::KdTree<pcl::PointXYZRGB>);

    FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, FPFHSignature33> fpfhEst;
    fpfhEst.setInputCloud (cloud.makeShared());
    fpfhEst.setInputNormals (normals);
    fpfhEst.setSearchMethod (kdTree);
    fpfhEst.setRadiusSearch (FEATURES_RADIUS);
    fpfhEst.compute (*features);
    
    return features;
}

// FPFH 2.0, version with only input cloud
pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFeatures(pcl::PointCloud<pcl::PointXYZRGB> cloud){
    
    // Features
    cout<<"Computing features..."<<endl;
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(cloud);
    
    PointCloud<FPFHSignature33>::Ptr features (new PointCloud<FPFHSignature33>);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    
    FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, FPFHSignature33> fpfhEst;
    fpfhEst.setInputCloud (cloud.makeShared());
    fpfhEst.setInputNormals (normals);
    fpfhEst.setSearchMethod (kdTree);
    fpfhEst.setRadiusSearch (FEATURES_RADIUS);
    fpfhEst.compute (*features);
    
    return features;
}

// Sample consensus - Initial alignment algo.
pcl::PointCloud<pcl::PointXYZRGB> sac_ia_align(pcl::PointCloud<pcl::PointXYZRGB> sourceCloud, pcl::PointCloud<pcl::PointXYZRGB> targetCloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr sourceFeatures, pcl::PointCloud<pcl::FPFHSignature33>::Ptr targetFeatures){
    
    // SAC - IA Alignment
    cout<<"Starting SAC-IA registration..."<<endl;
    
    pcl::PointCloud<pcl::PointXYZRGB> sourceCloud_registered;
    SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, FPFHSignature33> sac_ia;
    
    // input source and target clouds
    sac_ia.setInputSource(sourceCloud.makeShared());
    sac_ia.setSourceFeatures(sourceFeatures);
    sac_ia.setInputTarget(targetCloud.makeShared());
    sac_ia.setTargetFeatures(targetFeatures);
    
    // set the parameters
    sac_ia.setMaxCorrespondenceDistance (SAC_MAX_CORRESPONDENCE_DIST); // needs tune
    sac_ia.setMaximumIterations (MAX_SACIA_ITE); // needs tune
    sac_ia.setMinSampleDistance(MIN_SAMPLE_DISTANCE);
    
    // align process
    sac_ia.align(sourceCloud_registered);
    
    if (sac_ia.hasConverged()) {
        cout << "SAC-IA Fittness score: " << sac_ia.getFitnessScore() << endl;
        cout << "Final transform: " << endl;
        cout << sac_ia.getFinalTransformation() << endl;
    } else {
        cout << "SAC-IA Dont't converge!" << endl;
    }
    
    return sourceCloud_registered;
}


// SAC-IA 2.0, version with only input clouds
pcl::PointCloud<pcl::PointXYZRGB> sac_ia_align(pcl::PointCloud<pcl::PointXYZRGB> sourceCloud, pcl::PointCloud<pcl::PointXYZRGB> targetCloud){
    
    // compute features and normals behind
    pcl::PointCloud<pcl::Normal>::Ptr sourceNormals = computeNormals(sourceCloud);
    pcl::PointCloud<pcl::Normal>::Ptr targetNormals = computeNormals(targetCloud);

    PointCloud<FPFHSignature33>::Ptr sourceFeatures = computeFeatures(sourceCloud, sourceNormals);
    PointCloud<FPFHSignature33>::Ptr targetFeatures = computeFeatures(targetCloud, targetNormals);
    
    // SAC - IA Alignment
    cout<<"Starting SAC-IA registration..."<<endl;
    
    pcl::PointCloud<pcl::PointXYZRGB> sourceCloud_registered;
    SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, FPFHSignature33> sac_ia;
    
    // input source and target clouds
    sac_ia.setInputSource(sourceCloud.makeShared());
    sac_ia.setSourceFeatures(sourceFeatures);
    sac_ia.setInputTarget(targetCloud.makeShared());
    sac_ia.setTargetFeatures(targetFeatures);
    
    // set the parameters
    sac_ia.setMaxCorrespondenceDistance (SAC_MAX_CORRESPONDENCE_DIST); // needs tune
    sac_ia.setMaximumIterations (MAX_SACIA_ITE); // needs tune
    sac_ia.setMinSampleDistance(MIN_SAMPLE_DISTANCE);

    // align process
    cout<<"Starting sac_ia.align()..."<<endl;
    sac_ia.align(sourceCloud_registered);
    
    if (sac_ia.hasConverged()) {
        cout << "SAC-IA Fittness score: " << sac_ia.getFitnessScore() << endl;
        cout << "Final transform: " << endl;
        cout << sac_ia.getFinalTransformation() << endl;
    } else {
        cout << "SAC-IA Dont't converge!" << endl;
    }
    
    return sourceCloud_registered;
}

// SAC-IA 3.0, output transformation matrix
Eigen::Matrix4f sac_ia_transMatrix(pcl::PointCloud<pcl::PointXYZRGB> sourceCloud, pcl::PointCloud<pcl::PointXYZRGB> targetCloud){
    
    Eigen::Matrix4f initTransform; // (Eigen::Matrix4f::Identity())
    
    // compute features and normals behind
    pcl::PointCloud<pcl::Normal>::Ptr sourceNormals = computeNormals(sourceCloud);
    pcl::PointCloud<pcl::Normal>::Ptr targetNormals = computeNormals(targetCloud);
    
    PointCloud<FPFHSignature33>::Ptr sourceFeatures = computeFeatures(sourceCloud, sourceNormals);
    PointCloud<FPFHSignature33>::Ptr targetFeatures = computeFeatures(targetCloud, targetNormals);
    
    // SAC - IA Alignment
    cout<<"Starting SAC-IA registration..."<<endl;
    
    pcl::PointCloud<pcl::PointXYZRGB> sourceCloud_registered;
    SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, FPFHSignature33> sac_ia;
    
    // input source and target clouds
    sac_ia.setInputSource(sourceCloud.makeShared());
    sac_ia.setSourceFeatures(sourceFeatures);
    sac_ia.setInputTarget(targetCloud.makeShared());
    sac_ia.setTargetFeatures(targetFeatures);
    
    // set the parameters
    sac_ia.setMaxCorrespondenceDistance (SAC_MAX_CORRESPONDENCE_DIST); // needs tune
    sac_ia.setMaximumIterations (MAX_SACIA_ITE); // needs tune
    sac_ia.setMinSampleDistance(MIN_SAMPLE_DISTANCE);
    
    // align process
    cout<<"Starting sac_ia.align()..."<<endl;
    sac_ia.align(sourceCloud_registered);
    
    if (sac_ia.hasConverged()) {
        cout << "SAC-IA Fittness score: " << sac_ia.getFitnessScore() << endl;
        cout << "Final transform: " << endl;
        cout << sac_ia.getFinalTransformation() << endl;
    } else {
        cout << "SAC-IA Dont't converge!" << endl;
    }
    initTransform = sac_ia.getFinalTransformation();
    
    return initTransform;
}

// SAC-IA 4.0, output transformation matrix
Eigen::Matrix4f sac_ia_transMatrix(pcl::PointCloud<pcl::PointXYZRGB> sourceCloud, pcl::PointCloud<pcl::PointXYZRGB> targetCloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr sourceFeatures, pcl::PointCloud<pcl::FPFHSignature33>::Ptr targetFeatures){
    
    Eigen::Matrix4f initTransform; // (Eigen::Matrix4f::Identity())
    
    // SAC - IA Alignment
    cout<<"Starting SAC-IA registration..."<<endl;
    
    pcl::PointCloud<pcl::PointXYZRGB> sourceCloud_registered;
    SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, FPFHSignature33> sac_ia;
    
    // input source and target clouds
    sac_ia.setInputSource(sourceCloud.makeShared());
    sac_ia.setSourceFeatures(sourceFeatures);
    sac_ia.setInputTarget(targetCloud.makeShared());
    sac_ia.setTargetFeatures(targetFeatures);
    
    // set the parameters
    sac_ia.setMaxCorrespondenceDistance (SAC_MAX_CORRESPONDENCE_DIST); // needs tune
    sac_ia.setMaximumIterations (MAX_SACIA_ITE); // needs tune
    sac_ia.setMinSampleDistance(MIN_SAMPLE_DISTANCE);
    
    // align process
    cout<<"Starting sac_ia.align()..."<<endl;
    sac_ia.align(sourceCloud_registered);
    
    // Print the alignment fitness score (values less than 0.00002 are good)
    if (sac_ia.hasConverged()) {
        cout << "SAC-IA Fittness score: " << sac_ia.getFitnessScore() << endl;
        cout << "Final transform: " << endl;
        cout << sac_ia.getFinalTransformation() << endl;
    } else {
        cout << "SAC-IA Dont't converge!" << endl;
    }
    initTransform = sac_ia.getFinalTransformation();
    
    return initTransform;
}

// Iterative closest point alignment
// register aligned cloud with target cloud (from SAC-IA)
pcl::PointCloud<pcl::PointXYZRGB> icp_align(pcl::PointCloud<pcl::PointXYZRGB> alignedCloud, pcl::PointCloud<pcl::PointXYZRGB> targetCloud){
    
    cout<<"Starting ICP registration..."<<endl;
    
    pcl::PointCloud<pcl::PointXYZRGB> sourceCloud_registered;
    
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    
    icp.setInputSource (alignedCloud.makeShared()); // from resulting cloud of SAC-IA
    icp.setInputTarget (targetCloud.makeShared());
    
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance (ICP_MAX_CORRESPONDENCE_DIST);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations (MAX_ICP_ITE);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon (ICP_TRANSFORMATION_EPSILON);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon (ICP_EUCLIDEAN_FITNESS_EPSILON);
    //icp.setRANSACOutlierRejectionThreshold (distance);
    
    // align process
    cout<<"Starting icp.align()..."<<endl;
    icp.align (sourceCloud_registered);

    // Obtain the transformation that aligned cloud_source to cloud_source_registered
    // Eigen::Matrix4f transformation = icp.getFinalTransformation();
    
    if (icp.hasConverged()) {
        cout << "ICP Fittness score: " << icp.getFitnessScore() << endl;
        cout << "Final transform: " << endl;
        cout << icp.getFinalTransformation() << endl;
    } else {
        cout << "ICP Dont't converge!" << endl;
    }
    return sourceCloud_registered;
}

// ICP 2.0, output transformation matrix onto reference coordinate system
Eigen::Matrix4f icp_transMatrix(pcl::PointCloud<pcl::PointXYZRGB> alignedCloud, pcl::PointCloud<pcl::PointXYZRGB> targetCloud){
    
    Eigen::Matrix4f finalTransform = Eigen::Matrix4f::Identity (), prev;
    cout<<"Starting ICP registration..."<<endl;
    
    pcl::PointCloud<pcl::PointXYZRGB> sourceCloud_registered;
    
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree1 (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree1->setInputCloud(alignedCloud.makeShared());
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree2->setInputCloud(targetCloud.makeShared());
    icp.setSearchMethodSource(tree1);
    icp.setSearchMethodTarget(tree2);
    
    icp.setInputSource (alignedCloud.makeShared()); // from resulting cloud of SAC-IA
    icp.setInputTarget (targetCloud.makeShared());
    
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
//    icp.setMaxCorrespondenceDistance (ICP_MAX_CORRESPONDENCE_DIST);
//    // Set the maximum number of iterations (criterion 1)
//    icp.setMaximumIterations (MAX_ICP_ITE);
//    // Set the transformation epsilon (criterion 2)
//    icp.setTransformationEpsilon (ICP_TRANSFORMATION_EPSILON);
//    // Set the euclidean distance difference epsilon (criterion 3)
//    icp.setEuclideanFitnessEpsilon (ICP_EUCLIDEAN_FITNESS_EPSILON);
   
    // align process
//    icp.align (sourceCloud_registered);
//    finalTransform = icp.getFinalTransformation();
    
    
    icp.setTransformationEpsilon (1e-10);
    icp.setMaximumIterations (1);
    icp.setMaxCorrespondenceDistance (0.1);
    
    pcl::PointCloud<pcl::PointXYZRGB> reg_result = alignedCloud;
    int itrTime = 100;
    for(int i = 0; i<itrTime; i++){
       
        alignedCloud = reg_result;
        
        icp.setInputSource(alignedCloud.makeShared());
        icp.align(reg_result);
        
        finalTransform = icp.getFinalTransformation () * finalTransform;

        if (fabs((icp.getLastIncrementalTransformation () - prev).sum ())<icp.getTransformationEpsilon())
            icp.setMaxCorrespondenceDistance (icp.getMaxCorrespondenceDistance () / 1.01);
        
        prev = icp.getLastIncrementalTransformation ();
                
    }
//    transformPointCloud(alignedCloud, alignedCloud, finalTransform);
//    cout<<"mse: "<<computeCloudRMS(targetCloud.makeShared(), alignedCloud.makeShared(),  0.1);
    
    if (icp.hasConverged()) {
        cout << "ICP Fittness score: " << icp.getFitnessScore() << endl;
        cout << "Final transform: " << endl;
        cout << icp.getFinalTransformation() << endl;
    } else {
        cout << "ICP Dont't converge!" << endl;
    }
    
    return finalTransform;
}



pcl::PointCloud<pcl::PointXYZRGB> pcmerge(pcl::PointCloud<pcl::PointXYZRGB> cloud_ref, const pcl::PointCloud<pcl::PointXYZRGB> cloud_aligned, float mergesize) {
    
    pcl::KdTreeFLANN<pcl::PointXYZRGB> tree;
    tree.setInputCloud(cloud_ref.makeShared());
    std::vector<int> index;
    std::vector<float> dist;
    for(int k=0; k < cloud_aligned.size(); k++) {
        //avoid adding redundant points
        if( tree.radiusSearch(cloud_aligned[k],0.005,index,dist,1) == 0 ) {
            cloud_ref.push_back(cloud_aligned[k]);
        }
    }
    
//    cloud_ref = cloud_ref + cloud_aligned;
//    
//    std::cerr << "PointCloud after merge & before filtering: " << cloud_ref.width * cloud_ref.height
//    << " data points (" << pcl::getFieldsList (cloud_ref) << ")."<<endl;
//    // downsample
//    pcl::VoxelGrid<pcl::PointXYZRGB> vox;
//    vox.setInputCloud (cloud_ref.makeShared());
//    vox.setLeafSize (mergesize, mergesize, mergesize);
//    vox.filter (cloud_ref);
//    
//    std::cerr << "PointCloud after merge & after filtering: " << cloud_ref.width * cloud_ref.height
//    << " data points (" << pcl::getFieldsList (cloud_ref) << ")."<<endl;
    
    return cloud_ref;
}

Eigen::Matrix4f pairwiseRegister(pcl::PointCloud<pcl::PointXYZRGB> src, pcl::PointCloud<pcl::PointXYZRGB> tgt, bool pass_through = true, bool downsample = true, bool remove_outlier = true, bool resample = false){
    
    //pcl::PointCloud<pcl::PointXYZRGB> registered_src;
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity (), icpTransform;
    double elapsed_secs;
    clock_t begin, end;
    
    // preprocess/filtering
    // pass through filter
    if(pass_through){
        begin = clock();
        
        src = pass_filter(src);
        tgt = pass_filter(tgt);
        
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout<<"pass through time: "<<elapsed_secs<<"s"<<endl;
    }
    
    // downsample
    if(downsample){
        begin = clock();

        src = voxel_filter(src);
        tgt = voxel_filter(tgt);
        
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout<<"voxel time: "<<elapsed_secs<<"s"<<endl;

    }
    
    // remove outlier
    if(remove_outlier){
        begin = clock();

        src = outlier_filter(src);
        tgt = outlier_filter(tgt);
        
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout<<"outlier time: "<<elapsed_secs<<"s"<<endl;

    }
    
    if(resample){
        src = mls_resample(src);
        tgt = mls_resample(tgt);
    }
    
    // SAC-IA algo
    begin = clock();

    transformation = sac_ia_transMatrix(src, tgt);
    transformPointCloud(src, src, transformation);
    
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"sac-ia time: "<<elapsed_secs<<"s"<<endl;

    // ICP algo
    // order of product of transformation matrix
    begin = clock();

    icpTransform = icp_transMatrix(src, tgt);
    transformation = icpTransform * transformation;
    
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"icp time: "<<elapsed_secs<<"s"<<endl;

    return transformation;
}


#endif /* registration_h */
