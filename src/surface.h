//
//  surface.h
//  3dImage
//
//  Created by DiaMond on 7/30/16.
//  Copyright © 2016 DiaMond. All rights reserved.
//

#ifndef surface_h
#define surface_h

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
//#include <pcl/surface/marching_cubes_greedy.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/search.h>
#include <iostream>
#include <pcl/surface/gp3.h>

const float default_iso_level = 0.0f;
const float default_extend_percentage = 0.3f;
const int default_grid_res = 30;
const float default_off_surface_displacement = 0.01f;

using namespace pcl;
using namespace std;

// MLS
pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
    
    MovingLeastSquares<PointXYZRGB, PointXYZRGB> mls;
    
    mls.setInputCloud (cloud);
    mls.setSearchRadius (0.03);
    mls.setPolynomialFit (true);
    mls.setPolynomialOrder (2);
    //mls.setSqrGaussParam(0.0009);
    mls.setUpsamplingMethod (MovingLeastSquares<PointXYZRGB, PointXYZRGB>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius (0.005);
    mls.setUpsamplingStepSize (0.003);
    
    PointCloud<PointXYZRGB>::Ptr cloud_smoothed (new PointCloud<PointXYZRGB> ());
    mls.process (*cloud_smoothed);
    
    cout<<"point cloud after mls:"<<(*cloud_smoothed).points.size()<<endl;

    return cloud_smoothed;
}

// MLS version 2
pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothCloud2(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
    
    PointCloud<PointXYZRGB>::Ptr cloud_smoothed (new PointCloud<PointXYZRGB> ());
    
    MovingLeastSquares<PointXYZRGB, PointXYZRGB> mls;
    
    mls.setInputCloud (cloud);
    mls.setSearchRadius (4);
    mls.setPolynomialFit (true);
    mls.setPolynomialOrder (1);
    mls.setUpsamplingMethod (MovingLeastSquares<PointXYZRGB, PointXYZRGB>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius (1);
    mls.setUpsamplingStepSize (0.3);
    
    mls.process (*cloud_smoothed);

    cout<<"point cloud after mls:"<<(*cloud_smoothed).points.size()<<endl;

    return cloud_smoothed;
}

// compute surface normals for the cloud
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
    
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    
    // Normals
    cout<<"Computing normals..."<<endl;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(cloud);

    pcl::NormalEstimation<PointXYZRGB, Normal> normEst;
    
    // Set parameters
    normEst.setInputCloud (cloud);
    normEst.setSearchMethod (kdTree);
    normEst.setKSearch(20);
    normEst.compute (*normals);
    
    cout<<"normal size"<<(*normals).size()<<endl;
    return normals;
}

// normals version 2
pcl::PointCloud<pcl::Normal>::Ptr computeNormals2(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
    
    // Normals
    cout<<"Computing normals..."<<endl;
    
    NormalEstimationOMP<PointXYZRGB, Normal> ne;
    ne.setNumberOfThreads (8);
    ne.setInputCloud (cloud);
    ne.setRadiusSearch (0.8);
    Eigen::Vector4f centroid;
    compute3DCentroid (*cloud, centroid);
    ne.setViewPoint (centroid[0], centroid[1], centroid[2]);
    
    PointCloud<Normal>::Ptr cloud_normals (new PointCloud<Normal> ());
    ne.compute (*cloud_normals);
    
    for (size_t i = 0; i < cloud_normals->size (); ++i)
    {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
    }
    
    return cloud_normals;
}

// Marching cubes algorithm
pcl::PolygonMesh::Ptr mcReconstruct(pcl::PointCloud<PointXYZRGB>::Ptr pointcloud)
{
    std::cout << "start marching cubes algo." << std::endl;
    
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    
    pointcloud = smoothCloud(pointcloud);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(pointcloud);
    pcl::concatenateFields (*pointcloud, *normals, *cloud);

    pcl::search::KdTree<PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<PointXYZRGBNormal>);
    tree->setInputCloud(cloud);
    pcl::io::savePLYFile("output/cloud_normals.ply", *cloud);
    
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);
    
    pcl::MarchingCubesHoppe<PointXYZRGBNormal> mc;
    
    mc.setIsoLevel (default_iso_level);
    mc.setGridResolution (default_grid_res, default_grid_res, default_grid_res);
    mc.setPercentageExtendGrid (default_extend_percentage);
    mc.setSearchMethod(tree);
    mc.setInputCloud (cloud);
    
    mc.reconstruct(*triangles);
    
    return triangles;
}

// Greedy Projection Triangulation
pcl::PolygonMesh::Ptr gp3Reconstruct(pcl::PointCloud<PointXYZRGB>::Ptr pointcloud)
{
    std::cout << "start gp3 algo." << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    
    //pointcloud = smoothCloud(pointcloud);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(pointcloud);
    pcl::concatenateFields (*pointcloud, *normals, *cloud);
    
    
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree->setInputCloud(cloud);
    pcl::io::savePLYFile("output/cloud_normals.ply", *cloud);

    
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);
    
    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
    
    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.025);
    
    // Set typical values for the parameters
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);
    
    // Get result
    gp3.setInputCloud (cloud);
    gp3.setSearchMethod (tree);
    gp3.reconstruct (*triangles);
    
    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();
    
    std::cout << "finish gp3 algo." << std::endl;

    return triangles;
}

// Poisson Reconstruction
pcl::PolygonMesh::Ptr pnReconstruct(pcl::PointCloud<PointXYZRGB>::Ptr pointcloud){
    
    std::cout << "start poisson rec algo." << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    
    //pointcloud = smoothCloud(pointcloud);
    
    cout<<"point cloud after mls:"<<(*pointcloud).points.size()<<endl;
    
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(pointcloud);
    pcl::concatenateFields (*pointcloud, *normals, *cloud);
    
    //create search tree
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree->setInputCloud(cloud);
    
    pcl::Poisson<pcl::PointXYZRGBNormal> pn;
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);
    
    pn.setManifold(false) ;
    pn.setOutputPolygons(false) ;
    pn.setConfidence(false);
    pn.setIsoDivide(8);
    pn.setSamplesPerNode(3) ;
    //pn.setDepth(9);
    
    pn.setInputCloud(cloud);
    pn.setSearchMethod(tree);
    pn.reconstruct(*triangles);

    std::cout << "finish poisson reconstruction algo." << std::endl;

    return triangles;
}
#endif /* surface_h */
