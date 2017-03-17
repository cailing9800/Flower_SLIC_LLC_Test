#include "dictionary.h"

using namespace std;
using namespace cv;

//利用opencv自带的BOWKMeansTrainer进行KMeans聚类
void llc_gen_dictionary(const string& dictionaryFile, Mat &dictionary)
{
	FileStorage fs(dictionaryFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["dictionary"] >> dictionary;
	}
}
