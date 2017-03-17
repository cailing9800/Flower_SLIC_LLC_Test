#include "llc.h"

using namespace std;
using namespace cv;

/*
 * Locality-constrained linear coding(LLC)算法
 *包括coding和pooing过程
 */

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, int step, cv::Mat &X, cv::Mat &Y)
{
    std::vector<int> t_x, t_y;
    for(int i = xgv.start; i <= xgv.end; i += step) t_x.push_back(i);
    for(int j = ygv.start; j <= ygv.end; j += step) t_y.push_back(j);

    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}


void calculateSiftXY_opencv(int width, int height, int patchSize, int binSize, int step, Mat &feaSet_x, Mat &feaSet_y)
{
	int remX = (width - patchSize) % step;
	int offsetX = floor(remX/2) + 1;
	int remY = (height - patchSize) % step;
	int offsetY = floor(remY/2) + 1;

	cv::Mat gridX, gridY, gridXX, gridYY;
	meshgrid(cv::Range(offsetX, width-patchSize+1), cv::Range(offsetY, height-patchSize+1), step, gridXX, gridYY);

	transpose(gridXX, gridX);
	transpose(gridYY, gridY);

	for(int i = 0; i < gridX.rows; i++)
	{
		for(int j = 0; j < gridX.cols; j++)
		{
			feaSet_x.at<float>(j+i*gridX.cols, 0) = gridX.ptr<int>(i)[j] + patchSize/2;
		}
	}

	for(int i = 0; i < gridY.rows; i++)
	{
		for(int j = 0; j < gridY.cols; j++)
		{
			feaSet_y.at<float>(j+i*gridY.cols, 0) = gridY.ptr<int>(i)[j] + patchSize/2;
		}
	}

	gridX.release();
	gridY.release();
	gridXX.release();
	gridYY.release();
}


void calculateSiftXY_vlfeat(int width, int height, int patchSize, int binSize, int step, Mat &feaSet_x, Mat &feaSet_y)
{
    float offsetX = 0.5*binSize*(4-1);
    float offsetY = 0.5*binSize*(4-1);

	cv::Mat gridX, gridY, gridXX, gridYY;
	meshgrid(cv::Range(offsetX, width-patchSize/2+1), cv::Range(offsetY, height-patchSize/2+1), step, gridXX, gridYY);

	transpose(gridXX, gridX);
	transpose(gridYY, gridY);

	for(int i = 0; i < gridX.rows; i++)
	{
		for(int j = 0; j < gridX.cols; j++)
		{
			feaSet_x.at<float>(j+i*gridX.cols, 0) = gridX.ptr<int>(i)[j];
		}
	}

	for(int i = 0; i < gridY.rows; i++)
	{
		for(int j = 0; j < gridY.cols; j++)
		{
			feaSet_y.at<float>(j+i*gridY.cols, 0) = gridY.ptr<int>(i)[j];
		}
	}

	gridX.release();
	gridY.release();
	gridXX.release();
	gridYY.release();
}

//llc approximation coding
void llc_coding(Mat &B, Mat &X, Mat &llcCodes, int knn)
{
    //find k nearest neighbors
    int nframe = X.rows; //X(前面提取的dsift特征)矩阵的行
    int nbase = B.rows; //B(字典)矩阵的行

	Mat XX, BB;
	//reduce矩阵变向量，相当于matlab中的sum
	cv::reduce(X.mul(X), XX, 1, CV_REDUCE_SUM, CV_32FC1);
	cv::reduce(B.mul(B), BB, 1, CV_REDUCE_SUM, CV_32FC1);

	//repeat相当于matlab中的repmat
	Mat D1 = cv::repeat(XX, 1, nbase);
    Mat Bt;
    transpose(B, Bt); //注意转置阵不能返回给原Mat本身
    Mat D2 = 2*X*Bt;
    Mat BBt;
    transpose(BB, BBt);
	Mat D3 = cv::repeat(BBt, nframe, 1);
    Mat D = D1 - D2 + D3;

	Mat SD(nframe, nbase, CV_16UC1);
	//对D所有行升序排列后，将索引赋给SD
    cv::sortIdx(D, SD, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

    Mat IDX(nframe, knn, CV_16UC1);
	//将SD的第i列赋值给IDX的第i列
	for (int i = 0; i < knn; i++)
	{
		SD.col(i).copyTo(IDX.col(i));
	}

	float beta = 1e-4;
	int nxcol = X.cols; //特征行

    Mat II = Mat::eye(knn, knn, CV_32FC1);

    llcCodes = Mat::zeros(nframe, nbase, CV_32FC1);

    Mat z, zt;
    Mat z1(knn, nxcol, CV_32FC1);
    Mat z2(knn, nxcol, CV_32FC1);

    Mat C, C_inv;
    Mat w, wt;

    for (int i = 0; i < nframe; i++)
	{
        for (int j = 0; j < knn; j++)
		{
			B.row(IDX.ptr<ushort>(i)[j]).copyTo(z1.row(j));
			X.row(i).copyTo(z2.row(j));
        }
        z = z1 - z2;

        transpose(z, zt);
        C = z*zt;
        C = C + II*beta*trace(C)[0]; //trace(C)[0]求矩阵的迹
        invert(C, C_inv);

        w = C_inv*Mat::ones(knn, 1, CV_32FC1); //相当于matlab中的w = C\ones(knn,1);

        float sum_w = 0;
		sum_w = cv::sum(w)[0];
        w = w/sum_w;
        transpose(w, wt);

        for (int j = 0; j < knn; j++)
		{
            llcCodes.at<float>(i, IDX.ptr<ushort>(i)[j]) = wt.at<float>(0, j);
        }
    }

    XX.release();
    BB.release();
	Bt.release();
	BBt.release();
    D.release();
    D1.release();
    D2.release();
	D3.release();
    SD.release();
    II.release();
    z.release();
	zt.release();
    z1.release();
    z2.release();
    C.release();
	C_inv.release();
    w.release();
    wt.release();
}

void llc_pooling(Mat &tdictionary,
                 Mat &tinput,
                 Mat &tllccodes,
                 Mat &llcFeature,
                 Mat &feaSet_x,
                 Mat &feaSet_y,
                 int width,
                 int height
                )
{
	Mat dictionary, input;
	transpose(tdictionary, dictionary);
	transpose(tinput, input);

	int dSize = dictionary.cols;
	int nSmp = input.cols;

	Mat idxBin = Mat::zeros(nSmp, 1, CV_32FC1);
	Mat llccodes;
	transpose(tllccodes, llccodes);

	Mat pyramid(1, 3, CV_32FC1);
	pyramid.at<float>(0, 0) = 1;
	pyramid.at<float>(0, 1) = 2;
	pyramid.at<float>(0, 2) = 4;

	int pLevels = pyramid.cols;
	Mat pBins(1,3,CV_32FC1);
	int tBins = 0;
	for(int i = 0; i < 3; i++)
	{
		pBins.at<float>(0, i) = pyramid.at<float>(0, i)*pyramid.at<float>(0, i);
		tBins += pBins.at<float>(0, i);
	}
	Mat beta = Mat::zeros(dSize, tBins, CV_32FC1);
	int bId = 0;
	int betacol = -1; //beta的列


	for (int iter1 = 0; iter1 != pLevels; iter1++)
	{
		int nBins = pBins.at<float>(0, iter1);
		float wUnit = width / pyramid.at<float>(0, iter1);
		float hUnit = height / pyramid.at<float>(0, iter1);

		//find to which spatial bin each local descriptor belongs
		Mat xBin(nSmp, 1, CV_32FC1);
		Mat yBin(nSmp, 1, CV_32FC1);

		for(int i = 0; i < nSmp; i++)
		{
			xBin.at<float>(i, 0) = ceil(feaSet_x.at<float>(i, 0) / wUnit);
		    yBin.at<float>(i, 0) = ceil(feaSet_y.at<float>(i, 0) / hUnit);
			idxBin.at<float>(i, 0) = (yBin.at<float>(i, 0) - 1) * pyramid.at<float>(0, iter1) + xBin.at<float>(i, 0);
		}

		for(int iter2 = 1; iter2 <= nBins; iter2++)
		{
			bId = bId + 1;
			betacol = betacol + 1;

			int nsbrows = 0; //统计每次循环sidxBin的行总数
			for(int i = 0; i < nSmp; i++)
			{
				if(idxBin.at<float>(i, 0) == iter2)
				{
					nsbrows++;
				}
			}

			Mat sidxBin(nsbrows, 1, CV_16UC1);
			int sbrow = 0; //sidxBin的行
			for(int i = 0; i < nSmp; i++)
			{
				if(idxBin.at<float>(i, 0) == iter2)
				{
					sidxBin.ptr<ushort>(sbrow++)[0] = i;
				}
			}
			if(sidxBin.empty())
			{
				continue;
			}

			//beta(:, bId) = max(llc_codes(:, sidxBin), [], 2);
			float iRowMax = 0; //每一行的最大值
			for(int i = 0; i < llccodes.rows; i++)
			{
				iRowMax = llccodes.at<float>(i, sidxBin.ptr<ushort>(0)[0]);
				for(int j = 0; j < nsbrows; j++)
				{
					if(llccodes.at<float>(i, sidxBin.ptr<ushort>(j)[0]) > iRowMax)
					{
						iRowMax = llccodes.at<float>(i, sidxBin.ptr<ushort>(j)[0]);
					}
				}
				beta.at<float>(i, betacol) = iRowMax;
			}
		}
	}

	if(bId != tBins)
	{
		cout<<"Index number error!"<<endl;
		exit;
	}

    llcFeature = Mat(dSize*tBins, 1,  CV_32FC1);

	for(int i = 0; i < tBins; i++)
	{
		for(int j = 0; j < dSize; j++)
		{
			llcFeature.at<float>(j+i*dSize, 0) = beta.at<float>(j, i);
		}
	}

	float sum = 0; //注意类型是float不是int
	for(int i = 0; i < dSize*tBins; i++)
	{
		sum += llcFeature.at<float>(i, 0) * llcFeature.at<float>(i, 0);
	}
	llcFeature = llcFeature/sqrt(sum);

	dictionary.release();
	input.release();
	idxBin.release();
	idxBin.release();
	llccodes.release();
	pyramid.release();
	pBins.release();
	beta.release();
	feaSet_x.release();
	feaSet_y.release();
}

void llc_coding_pooling(cv::Mat &img,
						cv::Mat &dsiftFeature,
						cv::Mat &dictionary,
						cv::Mat &llcFeature,
                        int knn,
                        int step,
                        int binSize,
                        int patchSize
                       )
{
	int width = img.cols;
	int height = img.rows;
	
	Mat llcCodes;

	Mat feaSet_x(dsiftFeature.rows, 1, CV_32FC1);
    Mat feaSet_y(dsiftFeature.rows, 1, CV_32FC1);

	calculateSiftXY_opencv(width, height, patchSize, binSize, step, feaSet_x, feaSet_y);
	//calculateSiftXY_vlfeat(width, height, patchSize, binSize, step, feaSet_x, feaSet_y);

	llc_coding(dictionary, dsiftFeature, llcCodes, knn);

    llc_pooling(dictionary, dsiftFeature, llcCodes, llcFeature, feaSet_x, feaSet_y, width, height);

	dsiftFeature.release();
	feaSet_x.release();
	feaSet_y.release();
	llcCodes.release();
	//llcFeature.release();
}
