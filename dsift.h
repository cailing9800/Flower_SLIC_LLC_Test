#ifndef DSIFT_H_INCLUDED
#define DSIFT_H_INCLUDED

#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"

extern "C" {
#include "vl/dsift.h"
}

//利用opencv自带的sift特征提取
void llc_extract_dsift_feature(cv::Mat &img, int step, int patchSize, cv::Mat &dsiftFeature);

//利用vlfeat库提取dsift特征
void llc_extract_dsift_feature_vlfeat(cv::Mat &img, int step, int binSize, cv::Mat &dsiftFeature);

#endif // DSIFT_H_INCLUDED
