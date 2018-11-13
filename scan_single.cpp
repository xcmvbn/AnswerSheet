#pragma once
#pragma execution_character_set("utf-8")
// 本文件为utf-8 编码格式

/**
Created on Mon Oct 29 23:24:07 2018

@environment:	 VS2013 + OpenCV3.1.0 
@author:	Jeff_Xiang
*/

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <windows.h>
#include <sstream>
#include <fstream>
#include <iostream>  
#include <vector>
#include <time.h>
#include <iomanip>

using namespace cv;
using namespace std;

int debug = 1;
int debug_output = 1;
int debug_enhance = 1;

void showTheCurrentTime();						//Show The Current Time

// Main Functions
Mat imagePreprocessing(Mat &img_src);			// Image Preprocessing
Mat getROIArea(Mat &src_image, Mat &img_canny); // Get ROI Area
Mat imageEnhance(Mat &img_undistort);			// Image Enhance

// Sub Functions
vector<Point> sortPoints(vector<Point> &cnts);	// Sort Points
Mat whiteBlance(Mat &img_src);					// White Balance
Mat logEnhance(Mat &img_src);					// Logarithm Enhance

												// Color Enhance
void find_vhi_vlo(const Mat &img, double &vhi, double &vlo);
void quantizing_v(Mat &img, double vMax, double vMin);
void color_enhance(const Mat &img, Mat &dst, double vMax, double vMin);

int main(int argc, char** argv){
	cout.setf(ios::fixed);  // set precision
	showTheCurrentTime();
	DWORD start_time = GetTickCount();			// Start Counting

//1. Load Image
	cv::String path = "E:/dip_class_pics/pics/IMG_9447.JPG";
	// Test:	IMG_9502	
	//cv::String path = "E:/dip_class_pics/pics_sample_test/IMG_9375.JPG";
	Mat src_image = imread(path);
	cv::resize(src_image, src_image, Size(src_image.cols/2, src_image.rows/2), INTER_AREA);
	if (src_image.empty()){
		cout << "Error opening orginal image!" << endl;
	}
	if (debug){
		namedWindow("src_image", CV_WINDOW_NORMAL);
		cvResizeWindow("src_image", src_image.cols / 3, src_image.rows / 3);
		imshow("src_image", src_image);
	}

//2. Image Preprocessing
	Mat img_canny = imagePreprocessing(src_image);
	//DWORD start_time = GetTickCount();			// Start Counting
//3. Find Contours in the edge Image    AND   Draw Corners   AND  Warp Perspective
	Mat img_undistort = getROIArea(src_image, img_canny);

//4. IMAGE Enhance
	Mat img_enhacne = imageEnhance(img_undistort);

	// Counting time stop
	cout << "picture is completly processed " << endl;
	cout << "##############################################" << endl;
	DWORD end_time = GetTickCount(); // Stop Counting
	cout << "The run time is:" << (end_time - start_time) << "ms!" << endl; // Output Time

	// Wait and Exit
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

void showTheCurrentTime(){
	time_t now;
	struct tm *fmt;
	time(&now);
	fmt = localtime(&now);
	cout << "##############################################" << endl;
	cout << "Current time is : " << endl;
	cout << fmt->tm_year + 1900 << " y  " << fmt->tm_mon + 1 << " m  " << fmt->tm_mday << " d  " << endl;
	cout << fmt->tm_hour << " h  " << fmt->tm_min << " m  " << fmt->tm_sec << " s " << endl;
	cout << "##############################################" << endl;
}


// Main Functions
Mat imagePreprocessing(Mat &img_src){
	
	// Convert BGR to Gray
	Mat img_gray;
	cvtColor(img_src, img_gray, CV_BGR2GRAY);  //turn BGR to GRAY
	//judge if img_gray is empty
	if (img_gray.empty()){
		cout << "Error opening gray image!" << endl;
	}

	// Preprocessing
	//Pre - processing
	//	1. Gaussian blur(Reduce noise by smoothing the image)
	//	2. Adaptive Thresholding for binary image
	//	3. Morphological Operation(CLOSE) to Fill holes and reduce clutter
	Mat img_blur, img_thresh;
	// 1. Gaussian Blus
	cv::GaussianBlur(img_gray, img_blur, Size(23, 23), 0, 0); 
	if (debug){
		namedWindow("img_blur", CV_WINDOW_NORMAL);
		cvResizeWindow("img_blur", img_blur.cols / 2, img_blur.rows / 2);
		imshow("img_blur", img_blur);
	}
	// 2. Adaptive Thresholding for binary image
	cv::adaptiveThreshold(img_blur, img_thresh, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 2);
	if (debug){
		namedWindow("img_thresh", CV_WINDOW_NORMAL);
		cvResizeWindow("img_thresh", img_thresh.cols / 2, img_thresh.rows / 2);
		imshow("img_thresh", img_thresh);
	}

	// 3. Morphological Operation(CLOSE) to Fill holes and reduce clutter
	Mat kernel;
	kernel = getStructuringElement(MORPH_RECT, Size(9, 9));  
	cv::morphologyEx(img_thresh, img_thresh, MORPH_OPEN, kernel);			// Open morphology
	if (debug){
		namedWindow("img_thresh_morph", CV_WINDOW_NORMAL);
		cvResizeWindow("img_thresh_morph", img_thresh.cols / 2, img_thresh.rows / 2);
		imshow("img_thresh_morph", img_thresh);
	}

	// Paper detection algorithm
	//1. find border with Canny Detector
	//2. find contours in the segmented binary image
	//3. approxPolyDP
	//		1. CannyDetector to find edges
	Mat img_canny;
	int low_shreshold = 40;
	int low_high_ratio = 5;
	int high_shreshold = low_shreshold * low_high_ratio;
	Canny(img_thresh, img_canny, low_shreshold, high_shreshold);
	if (debug){
		namedWindow("img_canny", CV_WINDOW_NORMAL);
		cvResizeWindow("img_canny", img_canny.cols / 2, img_canny.rows / 2);
		imshow("img_canny", img_canny);
	}

	cout << "Image is pre processed " << endl;
	cout << "##############################################" << endl;
	return img_canny;
}

Mat getROIArea(Mat &src_image, Mat &img_canny){
	// (1) FindContours in the edge Image
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img_canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Mat img_contours;
	// src_image.copyTo(img_contours);
	// int contours_size = contours.size();
	// for (int i = 0; i < contours_size; i++){
	// 	vector<Point> contour = contours[i];
	// 	double perimeter_con = arcLength(contour, true);
	// 	//double area_con = contourArea(contour);
	// 	vector<vector<Point>> approx_contours(1);
	// 	if (perimeter_con > 1000){
	// 	//if ((perimeter_con > 1000) && (area_con > 1000)){
	// 		approxPolyDP(contour, approx_contours[0], 0.04 * perimeter_con, true);
	// 		drawContours(img_contours, approx_contours, 0, Scalar(0, 255, 255), 2);
	// 	}
	// }
	// if (debug_output){
	// 	namedWindow("img_contours", CV_WINDOW_NORMAL);
	// 	cvResizeWindow("img_contours", img_contours.cols / 2, img_contours.rows / 2);
	// 	imshow("img_contours", img_contours);
	// }

	vector<vector<Point>> cnts = contours;

	vector <Point> docCnt;// contours after judge
	// Make sure at least one contour will be found
	if (cnts.size() > 0){
		int contours_size = cnts.size();
		for (int i = 0; i < contours_size; i++){
			vector<Point> cnt = cnts[i];
			//vector<vector<Point>> hull(1);
			//convexHull(cnt, hull[0], false);
			double area = contourArea(cnt);
			double perimeter = arcLength(cnt, true);
			double area_all = ((double)src_image.cols * src_image.rows);				//	12192768
			double perimeter_all = (((double)src_image.cols + src_image.rows) * 2);		//	14112
			double area_ratio = ((double)area / area_all);
			double perimeter_ratio = ((double)perimeter / perimeter_all);
			vector<vector<Point>> approxsub(1);

			// 6. Filter of extremely unlikely contours Generation
			if ((area_ratio >= 0.20 && area_ratio <= 0.75) && (perimeter_ratio >= 0.40 && perimeter_ratio <= 0.85)){
				Mat img_cnt;
				src_image.copyTo(img_cnt);
				// approxPolyDP: turn point set into anti-clockwise
				approxPolyDP(cnt, approxsub[0], 0.04 * perimeter, true);
				drawContours(img_cnt, approxsub, 0, Scalar(0, 255, 0), 2);
				// print out area and perimeter
				cout << "area_ratio = " << fixed << setprecision(4) << ((double)area / area_all) << endl;
				cout << "perimeter_ratio = " << fixed << setprecision(4) << ((double)perimeter / perimeter_all) << endl;
				cout << endl;
				// Draw input and output image(from Contours to Convexhull)
				if (debug_output){
					namedWindow("img_cnt", CV_WINDOW_NORMAL);
					cvResizeWindow("img_cnt", img_cnt.cols / 2, img_cnt.rows / 2);
					imshow("img_cnt", img_cnt);
				}

				if (approxsub[0].size() == 4 || approxsub[0].size() == 3){
					// Make sure that left-top point is first, and points are anti-countwised
					docCnt = sortPoints(approxsub[0]);
					break;
				}
			}
		}
	}
	// (2) Draw the corner on the original image
	int docCnt_size = docCnt.size();
	Mat img_colorp;
	src_image.copyTo(img_colorp);
	for (int i = 0; i < docCnt_size; i++){
		// draw four circles on original image
		circle(img_colorp, Point(docCnt[i].x, docCnt[i].y), 25, Scalar(0, 255, 0), -1);
		if (debug_output){
			namedWindow("img_colorp", CV_WINDOW_NORMAL);
			cvResizeWindow("img_colorp", img_colorp.cols / 2, img_colorp.rows / 2);
			imshow("img_colorp", img_colorp);
		}
	}

	// (3) Mapping the detected Paper from quadrilateral to rectangle(cropWarp)
	Point2f pts1[4];
	for (int i = 0; i < docCnt_size; i++){
		pts1[i] = docCnt[i];
	}
	Point2f pts2[4];
	pts2[0].x = 0;				 pts2[0].y = 0;
	pts2[1].x = 0;				 pts2[1].y = src_image.rows;
	pts2[2].x = src_image.cols;  pts2[2].y = src_image.rows;
	pts2[3].x = src_image.cols;	 pts2[3].y = 0;

	//Warp images
	Mat img_undistort;
	Mat m_transform = getPerspectiveTransform(pts1, pts2);
	warpPerspective(src_image, img_undistort, m_transform, Size(src_image.cols, src_image.rows));
	resize(img_undistort, img_undistort, Size(705, 955));		//141/191, answer area ratio
	if (debug_output){
		namedWindow("img_undistort", CV_WINDOW_NORMAL);
		cvResizeWindow("img_undistort", img_undistort.cols / 1.5, img_undistort.rows / 1.5);
		imshow("img_undistort", img_undistort);
	}

	cout << "Image is getROIArea processed " << endl;
	cout << "##############################################" << endl;
	return img_undistort;
}

Mat imageEnhance(Mat &img_undistort){
	// There is 4 steps to enhance:
	// 1. Get ROI area
	// 2. White Balance 
	// 3. Logarithm Enhance 
	// 4. Color Enhance 
	Mat img_enhacne = img_undistort(Range(70, 885), Range(1, img_undistort.cols));	// first rows, then cols
	if (debug_enhance)		{ imshow("img_enhacne", img_enhacne); }
	img_enhacne = whiteBlance(img_enhacne);											// White Balance
	if (debug_enhance)		{ imshow("white_balance", img_enhacne); }
	img_enhacne = logEnhance(img_enhacne);											// Logarithm Enhance
	if (debug_enhance)		{ imshow("logEnhance", img_enhacne); }

	double vhi, vlo;																// Color Enhance
	find_vhi_vlo(img_enhacne, vhi, vlo);
	Mat img_colorEnhance(img_enhacne.size(), img_enhacne.type());
	color_enhance(img_enhacne, img_colorEnhance, vhi, vlo);
	if (debug_enhance){
		namedWindow("img_colorEnhance", CV_WINDOW_NORMAL);
		cvResizeWindow("img_colorEnhance", img_colorEnhance.cols / 1.5, img_colorEnhance.rows / 1.5);
		imshow("img_colorEnhance", img_colorEnhance);
	}

	cout << "Image is imageEnhance processed " << endl;
	cout << "##############################################" << endl;
	return img_enhacne;
}


// Sub Functions
vector<Point> sortPoints(vector<Point> &cnts){
	vector<Point> sorted_cnt(4); // outout pointset
	Point center;			  // center point
	int pointset_size = cnts.size();// calculate the x and y of center
	for (int i = 0; i < pointset_size; i++){
		center.x += cnts[i].x;
		center.y += cnts[i].y;
	}
	center.x = (double)(center.x / pointset_size);
	center.y = (double)(center.y / pointset_size);
	vector <Point> above;  // sort the points by up and down
	vector <Point> down;
	for (int i = 0; i < pointset_size; i++){
		if (cnts[i].y < center.y)	{ above.push_back(cnts[i]); }
		else						{ down.push_back(cnts[i]); }
	}
	// For the up points, left one is 1, right one is 4
	if (above[0].x < above[1].x)  { sorted_cnt[0] = above[0]; sorted_cnt[3] = above[1]; }
	else                          { sorted_cnt[0] = above[1]; sorted_cnt[3] = above[0]; }
	// For the down points, left one is 2, right one is 3
	if (down[0].x < down[1].x)	{ sorted_cnt[1] = down[0];	sorted_cnt[2] = down[1]; }
	else						{ sorted_cnt[1] = down[1];	sorted_cnt[2] = down[0]; }

	return sorted_cnt;
}

Mat whiteBlance(Mat &img_src){
	Mat img_dst; // output image
	vector<Mat> g_vChannels;
	//waitKey(0);

	//分离通道
	split(img_src, g_vChannels);
	vector<Mat> singleChannel(3);
	double imageChannelAvg[3];
	for (int i = 0; i < 3; i++){ // B -> G -> R
		singleChannel[i] = g_vChannels.at(i);
		imageChannelAvg[i] = mean(singleChannel[i])[0]; // get every channel average
	}
	
	//求出个通道所占增益
	double K = (imageChannelAvg[0] + imageChannelAvg[1] + imageChannelAvg[2]) / 3;
	double Kdata[3];
	for (int i = 0; i < 3; i++){ // B -> G -> R
		Kdata[i] = K / imageChannelAvg[i];
		addWeighted(singleChannel[i], Kdata[i], 0, 0, 0, singleChannel[i]);// refresh BGR after white balance
		//equalizeHist(singleChannel[i], singleChannel[i]);				// Histogram equalization
	}
	
	merge(g_vChannels, img_dst);//图像各通道合并

	return img_dst;
}

Mat logEnhance(Mat &img_src){
	Mat img_dst(img_src.size(), CV_32FC3);
	double temp = 255 / log(256);

	for (int i = 0; i < img_src.rows; i++){
		for (int j = 0; j < img_src.cols; j++){
			img_dst.at<Vec3f>(i, j)[0] = temp*log(1 + img_src.at<Vec3b>(i, j)[0]);
			img_dst.at<Vec3f>(i, j)[1] = temp*log(1 + img_src.at<Vec3b>(i, j)[1]);
			img_dst.at<Vec3f>(i, j)[2] = temp*log(1 + img_src.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255    
	normalize(img_dst, img_dst, 0, 255, CV_MINMAX);
	//转换成8bit图像显示    
	convertScaleAbs(img_dst, img_dst);

	return img_dst;
}

	//计算极大极小值
void find_vhi_vlo(const Mat &img, double &vhi, double &vlo){
	int width = img.cols;
	int height = img.rows;

	uchar min;
	vector<Mat> BGR;
	split(img, BGR);
	//conver to CMY
	BGR[0] = 255 - BGR[0];
	BGR[1] = 255 - BGR[1];
	BGR[2] = 255 - BGR[2];

	Mat CMY(img.size(), CV_8UC3);
	merge(BGR, CMY);

	for (int i = 0; i < height; i++){
		Vec3b *pImg = CMY.ptr<Vec3b>(i);
		for (int j = 0; j < width; j++){
			min = pImg[j][0];
			if (pImg[j][1] < min) min = pImg[j][1];
			if (pImg[j][2] < min) min = pImg[j][2];
			for (int k = 0; k < 3; k++){
				pImg[j][k] -= min;
			}
		}
	}
	Mat HSV = Mat(CMY.size(), CV_8UC3);
	cvtColor(CMY, HSV, COLOR_BGR2HSV);
	vector<Mat> vHSV;
	split(HSV, vHSV);

	////find Vmin and Vmax
	double vMin = 1.0;
	double vMax = .0;
	for (int i = 0; i < height; i++){
		uchar *pImg = vHSV[2].ptr<uchar>(i);
		for (int j = 0; j < width; j++){
			double v = (double)pImg[j] / 255.0;
			if (v > vMax)	vMax = v;
			if (v < vMin)	vMin = v;
		}
	}
	vhi = vMax;
	vlo = vMin;
}

	//v线性量化		img单通道
void quantizing_v(Mat &img, double vMax, double vMin){
	if (vMax == vMin)	return;
	if (img.channels() != 1)	return;

	double vdelta = 255;		// Color Enhance  parameter
	int width = img.cols;
	int height = img.rows;
	for (int i = 0; i < height; i++){
		uchar *pImg = img.ptr<uchar>(i);
		for (int j = 0; j < width; j++){
			double newPixel = ((double)pImg[j] / vdelta - vMin) / (vMax - vMin);
			int tmp = int(newPixel * 255);
			if (tmp >255)
				pImg[j] = 255;
			else if (tmp < 0)
				pImg[j] = 0;
			else
				pImg[j] = (uchar)tmp;
		}
	}
}

void color_enhance(const Mat &img, Mat &dst, double vMax, double vMin)
{
	double v;//v量化

	int width = img.cols;
	int height = img.rows;

	uchar min;
	vector<Mat> BGR;
	split(img, BGR);

	BGR[0] = 255 - BGR[0];
	BGR[1] = 255 - BGR[1];
	BGR[2] = 255 - BGR[2];

	Mat CMY(img.size(), CV_8UC3);
	merge(BGR, CMY);
	Mat minMat(img.size(), CV_8UC1);
	for (int i = 0; i < height; i++){
		Vec3b *pImg = CMY.ptr<Vec3b>(i);
		uchar *pMin = minMat.ptr<uchar>(i);

		for (int j = 0; j < width; j++){
			min = pImg[j][0];
			if (pImg[j][1] < min) min = pImg[j][1];
			if (pImg[j][2] < min) min = pImg[j][2];
			pMin[j] = min;
			for (int k = 0; k < 3; k++){
				pImg[j][k] -= min;
			}
		}
	}

	Mat HSV = Mat(CMY.size(), CV_8UC3);
	cvtColor(CMY, HSV, COLOR_BGR2HSV);

	vector<Mat> vHSV;
	split(HSV, vHSV);
	quantizing_v(vHSV[2], vMax, vMin);

	merge(vHSV, dst);

	cvtColor(dst, dst, COLOR_HSV2BGR);
	for (int i = 0; i < height; i++){
		Vec3b *pDst = dst.ptr<Vec3b>(i);
		uchar *pMin = minMat.ptr<uchar>(i);

		for (int j = 0; j < width; j++){
			for (int k = 0; k<3; k++){
				int tmp = pDst[j][k] + pMin[j];
				if (tmp > 255)
					pDst[j][k] = 0;
				else
					pDst[j][k] = 255 - tmp;
			}
		}
	}
}



