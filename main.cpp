/*
* Written by: wangshuang.
* LLC flower classification.
* 1. segmentation.
* 2. classification.
*/

#include "utils.h"
#include "slic.h"
#include "segment.h"
#include "dsift.h"
#include "dictionary.h"
#include "llc.h"
#include "predict.h"

using namespace std;
using namespace cv;

int step = 6;
int binSize = 4;
int patchSize = 16;
int maxImgSize = 300;
int knn = 5;

string dictionaryFile = "result/dictionary.xml.gz";
char modelFile[] = "result/model.txt";
char testFile[] = "result/test.txt";
char resultFile[] = "result/result.txt";
		
int main(int argc, char* argv[])
{
	const char *flower = "rose.jpg";
	vector<IplImage*> imageRect;
	vector<IplImage*> seg;
	
	slic_superpixels_segment(flower, imageRect, seg);
	// cout << imageRect.size() << endl;
	
	cv::Mat dictionary;
 	llc_gen_dictionary(dictionaryFile, dictionary);
 	
	for(int i = 0; i < seg.size(); i++)
    {  
    	cvShowImage("result", imageRect[i]);
        cv::Mat img(seg[i], 0);	
    	imshow("image", img);
    	
        llc_image_pretreatment(img, maxImgSize);
        
 		cv::Mat dsiftFeature;
 		llc_extract_dsift_feature(img, step, patchSize, dsiftFeature);
 		
 		cv::Mat llcFeature;
 		llc_coding_pooling(img, dsiftFeature, dictionary, llcFeature, 
 		                   knn, step, binSize, patchSize);
 		
    	llc_gen_txt_file(llcFeature, testFile);
		
		int labels = SVM_predict(argc, argv, testFile, modelFile, resultFile);
		cout << "predict label: " << labels << endl;
		waitKey();
	}
	
	return 0;
}
