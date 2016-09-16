//
//  data.hpp
//  3dImage
//
//  Created by DiaMond on 6/14/16.
//  Copyright © 2016 DiaMond. All rights reserved.
//

#ifndef data_hpp
#define data_hpp

#include <iostream>
#include <string>
#include <math.h>

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//include pcl
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>

using namespace cv;
using namespace std;
using namespace pcl;

const float dc1 = -0.0030711016;
const float dc2 = 3.3309495161;

// intrinsic parameters from camera calibration
const float fx_d = 591.16;
const float fy_d = 579.50;
const float px_d = 334.01;
const float py_d = 248.89;

const float unit_factor = 1000.0f;

const float fx_d2 = 5.9421434211923247e+02;
const float fy_d2 = 5.9104053696870778e+02;
const float px_d2 = 3.3930780975300314e+02;
const float py_d2 = 2.4273913761751615e+02;

const float k1_d = -2.6386489753128833e-01;
const float k2_d = 9.9966832163729757e-01;
const float p1_d = -7.6275862143610667e-04;
const float p2_d = 5.0350940090814270e-03;
const float k3_d = -1.3053628089976321e+00;

// output type of structure
string type2str(int type) {
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    r += "C";
    r += (chans+'0');
    
    return r;
}

// bilateral filter
Mat bilateral(Mat &depthMat, int size){
    Mat depthMat2, final;
    double min, max;
    cv::minMaxLoc(depthMat, &min, &max);
    
    depthMat.convertTo( depthMat2, CV_8UC1, 255.0/max);
    bilateralFilter(depthMat2, final, size, size*2, size/2);
    
    final.convertTo(final, CV_16UC1, max/255.0);
    
    return final;
}

Mat median(Mat &depthMat, int size){
    Mat depthMat2, final;
    double min, max;
    cv::minMaxLoc(depthMat, &min, &max);
    
    depthMat.convertTo( depthMat2, CV_8UC1, 255.0/max);
    medianBlur(depthMat2, final, size);
    
    final.convertTo(final, CV_16UC1, max/255.0);
    
    return final;
}

Mat gaussian(Mat &depthMat, int size){
    Mat depthMat2, final;
    double min, max;
    cv::minMaxLoc(depthMat, &min, &max);
    
    depthMat.convertTo( depthMat2, CV_8UC1, 255.0/max);
    GaussianBlur( depthMat2, final, Size( size, size ), 0, 0);
    
    final.convertTo(final, CV_16UC1, max/255.0);
    
    return final;
}

// mean squared error
double mse(Mat ref, Mat filtered){
    double error = 0;
    double sum = 0;
    int no = 0;
    if(ref.size()!=filtered.size()){
        cout<<"Different image size."<<endl;
        return -1;
    }
    
    for(int i = 0; i<ref.rows; i++){
        for(int j = 0; j<ref.cols; j++){
            int refVal =  (int) ref.at<uchar>(i,j);
            int filVal =  (int) filtered.at<uchar>(i,j);
            
            sum += pow((refVal - filVal), 2.0);
        }
    }
    
    error = sum/(ref.rows*ref.cols);
    return error;
}

// psnr
double psnr(double mse){
    
    return 10*log10(pow(255.0, 2.0)/mse);
}


void loadData(string str, vector<pcl::PointCloud<pcl::PointXYZRGB>> &models){
    string extension = ".pcd";
    int begin = 2;
    for(;;){
        pcl::PointCloud<pcl::PointXYZRGB> model;
        if( pcl::io::loadPCDFile<pcl::PointXYZRGB> (str+to_string(begin)+extension, model) == -1){
            
            cout<<"cannot read data file"<<endl;
            break;
        } else{
            // remove NAN points from the cloud
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(model, model, indices);
            models.push_back (model);
            begin++;
        }
    }
}

void loadPCD(vector<pcl::PointCloud<pcl::PointXYZRGB>> &models){
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    if( pcl::io::loadPCDFile<pcl::PointXYZRGB> ("/Users/DiaMond/Downloads/bun01.pcd", cloud) == -1){
        cout<<"cannot read data file"<<endl;
    }
    
    //remove NAN points from the cloud
    std::vector<int> indices1;
    pcl::removeNaNFromPointCloud(cloud, cloud, indices1);
    models.push_back (cloud);


    if( pcl::io::loadPCDFile<pcl::PointXYZRGB> ("/Users/DiaMond/Downloads/bun02.pcd", cloud) == -1){
        cout<<"cannot read data file"<<endl;
    }
    
    //remove NAN points from the cloud
    std::vector<int> indices2;
    pcl::removeNaNFromPointCloud(cloud, cloud, indices2);
    models.push_back (cloud);
}

// CV_16UC1 depth map
// try inpainting holes, losing details
void inpaint(Mat &depthMat, double &max){
    //interpolation & inpainting

    Mat depthf(Size(640,480),CV_8UC1);
    
    Mat tmp, tmp1;
    // shift the minimum distance
    Mat(depthMat - 500.0).convertTo(tmp1,CV_64FC1);
    
    double minval,maxval;
    minMaxLoc(tmp1, &minval, &maxval, NULL, NULL);
    max = maxval;
    tmp1.convertTo(depthf, CV_8UC1, 255.0/maxval);
    
    // smaller size for the ease of computation
    Mat small_depthf;
    resize(depthf,small_depthf,Size(),0.2,0.2);
    
    //inpaint only the holes (missing values)
    cv::inpaint(small_depthf,(small_depthf == 255),tmp1,5.0,INPAINT_TELEA);
    
    resize(tmp1, tmp, depthf.size());
    
    //add the original signal back over the inpaint
    tmp.copyTo(depthf, (depthf == 255));
    depthMat = depthf;
}

// right way to read CV_16UC1 depth image
//depthMap = imread(str+"depth_5.png", CV_LOAD_IMAGE_ANYDEPTH);
pcl::PointCloud<pcl::PointXYZRGB> computeCloudFromImg(Mat& depth_img, Mat& bgr_img){
    
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    //    cloud.width = depth_img.cols*depth_img.rows;
    //    cloud.height = 1;
    //    cloud.resize(cloud.width*cloud.height);
    
    cloud.width = depth_img.cols;
    cloud.height = depth_img.rows;
    cloud.resize(cloud.width*cloud.height);
    cloud.is_dense = false;
    
    if(depth_img.empty()){
        cout<<"no data captured"<<endl;
        return cloud;
    }
    cout<<"data type:"<<type2str(depth_img.type())<<endl;
    int no = 0;
    int zerodepth = 0;
    double min, max;
    cv::minMaxLoc(depth_img, &min, &max);
    cout<<max<<"; "<<min<<endl;
    
    for (int i = 0; i<depth_img.rows; i++) {
        for (int j = 0; j<depth_img.cols; j++) {
            Vec3b color = bgr_img.at<Vec3b>(i,j);
            double z = (double) depth_img.at<uchar>(i,j);
            
            pcl::PointXYZRGB p;
            
            if( !isnan(z)){      //depth is number
                if(z<=0){
                    zerodepth++;
                    continue;
                }
                p.z = z/255.0*10001.0;
//                cout<<p.z<<endl;
                p.x = p.z*(j-px_d)/fx_d;
                p.y = p.z*(i-py_d)/fy_d;
                
                // 3 channels - bgr
                p.b = color[0];
                p.g = color[1];
                p.r = color[2];
                
                //cloud.points.push_back (p);
                cloud(j,i) = p;
            }
            else{  //depth is a NaN
                pcl::PointXYZRGB p;
                p.z = std::numeric_limits<float>::quiet_NaN();
                p.x = std::numeric_limits<float>::quiet_NaN();
                p.y = std::numeric_limits<float>::quiet_NaN();
                p.r = std::numeric_limits<float>::quiet_NaN();
                p.g = std::numeric_limits<float>::quiet_NaN();
                p.b = std::numeric_limits<float>::quiet_NaN();
                
                //                cloud.points.push_back (p);
                cloud(j,i) = p;
                no++;
            }
        }
    }
    
    cout<<"1. here is "<< no <<" NAN values of depth!"<<endl<<endl;
    cout<<"2. here is "<< zerodepth <<" zero values of depth!"<<endl<<endl;
    
    return cloud;
}


//P3D.x = (x_d - cx_d) * depth(x_d,y_d) / fx_d
//P3D.y = (y_d - cy_d) * depth(x_d,y_d) / fy_d
//P3D.z = depth(x_d,y_d)

//P3D' = R.P3D + T
//P2D_rgb.x = (P3D'.x * fx_rgb / P3D'.z) + cx_rgb
//P2D_rgb.y = (P3D'.y * fy_rgb / P3D'.z) + cy_rgb
pcl::PointCloud<pcl::PointXYZRGB> computeColoredCloud(Mat& depth_img, Mat& bgr_img){

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
//    cloud.width = depth_img.cols*depth_img.rows;
//    cloud.height = 1;
//    cloud.resize(cloud.width*cloud.height);
    
    cloud.width = depth_img.cols;
    cloud.height = depth_img.rows;
    cloud.resize(cloud.width*cloud.height);
    cloud.is_dense = false;
    
    if(depth_img.empty()){
        cout<<"no data captured"<<endl;
        return cloud;
    }
    cout<<"data type:"<<type2str(depth_img.type())<<endl;
    int no = 0;
    int zerodepth = 0;
    double min, max;
    cv::minMaxLoc(depth_img, &min, &max);
    cout<<max<<"; "<<min<<endl;
//    8-bit unsigned integer (uchar)
//    16-bit unsigned integer (ushort)
//    32-bit floating-point number (float)
    
    for (int i = 0; i<depth_img.rows; i++) {
        for (int j = 0; j<depth_img.cols; j++) {
            Vec3b color = bgr_img.at<Vec3b>(i,j);
            ushort z = depth_img.at<ushort>(i,j);
            
            pcl::PointXYZRGB p;
            
            if( !isnan(z)){      //depth is number
                if(z<=0){
                    zerodepth++;
                    continue;
                }
                p.z = z/1000.0;
                p.x = p.z*(j-px_d)/fx_d;
                p.y = p.z*(i-py_d)/fy_d;
                
                // 3 channels - bgr
                p.b = color[0];
                p.g = color[1];
                p.r = color[2];
                
                //cloud.points.push_back (p);
                cloud(j,i) = p;
            }
            else{  //depth is a NaN
                pcl::PointXYZRGB p;
                p.z = std::numeric_limits<float>::quiet_NaN();
                p.x = std::numeric_limits<float>::quiet_NaN();
                p.y = std::numeric_limits<float>::quiet_NaN();
                p.r = std::numeric_limits<float>::quiet_NaN();
                p.g = std::numeric_limits<float>::quiet_NaN();
                p.b = std::numeric_limits<float>::quiet_NaN();
                
//                cloud.points.push_back (p);
                cloud(j,i) = p;
                no++;
            }
        }
    }
    
    cout<<"1. here is "<< no <<" NAN values of depth!"<<endl<<endl;
    cout<<"2. here is "<< zerodepth <<" zero values of depth!"<<endl<<endl;

    return cloud;
}

pcl::PointCloud<pcl::PointXYZ> computeXYZCloud(Mat& depth_img, Mat& bgr_img){

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = depth_img.cols;
    cloud.height = depth_img.rows;
    cloud.resize(cloud.width*cloud.height);
    for (int i = 0; i<depth_img.rows; i++) {
        for (int j = 0; j<depth_img.cols; j++) {
            //unsigned short z = depth_img.at<unsigned short>(i,j);
            float z = depth_img.at<int16_t>(i,j);
            cout<<"depth "<<z<<endl;
            if(z==0)
                continue;
            if(!isnan(z)){      //NaN is not equal to itself
                z = z/1000.0f; // mm->m
                cloud(j,i).x = z*(j-px_d)/fx_d;
                cloud(j,i).y = z*(i-py_d)/fy_d;
                cloud(j,i).z = z;
            }
            else{  //depth is a NaN
                cloud(j,i).z = std::numeric_limits<float>::quiet_NaN();
                cloud(j,i).x = std::numeric_limits<float>::quiet_NaN();
                cloud(j,i).y = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    return cloud;
}

//print cloud data (xyzbgr)
void printColoredCloud(PointCloud<PointXYZRGB> cloud){
    for(size_t i; i< cloud.size(); i++){
        cout<<cloud.points[i].x<<" "<<cloud.points[i].y<<" "<<cloud.points[i].z<<"; "<<cloud.points[i].b<<" "<<cloud.points[i].g<<" "<<cloud.points[i].r<<endl;
    }
}

// display rgb values of each pixel in bgr image
void displayRGBColors(Mat bgrImage){
    for (int i = 0; i<bgrImage.rows; i++) {
        for (int j = 0; j<bgrImage.cols; j++) {
            Vec3b color = bgrImage.at<Vec3b>(i,j);
            int b = color[0];
            int g = color[1];
            int r = color[2];
            cout<<"bgr color - b:"<<b<<"; g:"<<g<<"; r:"<<r<<"."<<endl;
        }
    }
}

void viewCloud(pcl::PointCloud<pcl::PointXYZRGB> cloud){
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(cloud.makeShared());
    while(!viewer.wasStopped());    //wasStopped() function processes mouse events for the window
}

#endif /* data_hpp */
