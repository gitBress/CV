#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>	
#include <opencv2/imgproc.hpp>
#include "filter.cpp"

using namespace cv;
using namespace std;


void showHistogram(std::vector<cv::Mat>& hists);
void on_trackbar_Gauss(int, void*);
void on_trackbar_Median(int, void*);
void on_trackbar_Bilateral(int, void*);
void Erode(int, void*);
void Dilate(int, void*);
void Morphology_Operations(int, void*);


//Global Variables

Mat finalLabImg;

int ksize;
int sigmaX;
int sigmaR;
int sigmaS;
GaussianFilter gauss = GaussianFilter(Mat(), 0, 0);
MedianFilter median = MedianFilter(Mat(), 0);
BilateralFilter bilateral = BilateralFilter(Mat(), 5, 0, 0);


int const max_operator = 1;
int const max_elem = 2;
int const max_kernel_size = 21;

Mat ImgErode;
int erosion_size;

Mat ImgDilate;
int dilate_size;

Mat ImgMorphEx;
int morph_elem;
int morph_size;
int morph_operator;




int main(int argc, char** argv) {

	Mat img = imread(".\\img\\image.jpg");

	namedWindow("Image", WINDOW_NORMAL);
	imshow("Image", img);

	//--------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------

	//HISTOGRAM EQUALIZATION

	//--------------------------------------------------------------------------------------------------------


	//RGB color space

	Mat bgr[3];			//destination array
	split(img, bgr);	//split source  
	
	Mat hist[3];
	int nbins = 256;	// bin size
	const int* histSize = &nbins;   
	float range[] = { 0, 255 };
	const float *ranges[] = { range };
	Mat equalImg[3];
	Mat equalHist[3];

	for (int i = 0; i < size(bgr); i++) {
		calcHist(&bgr[i], 1, 0, Mat(), hist[i], 1, histSize, ranges, true, false);
		equalizeHist(bgr[i], equalImg[i]);
		calcHist(&equalImg[i], 1, 0, Mat(), equalHist[i], 1, histSize, ranges, true, false);
	}
	
	vector<Mat> hists = { hist[0], hist[1], hist[2] };
	showHistogram(hists);
	
	waitKey(0); //to show the final result

	Mat outputImg;
	merge(equalImg, 3, outputImg);

	namedWindow("Image", WINDOW_NORMAL);
	imshow("Image", outputImg);

	vector<Mat> equalHists = { equalHist[0], equalHist[1], equalHist[2] };
	showHistogram(equalHists);

	waitKey(0); //to show converted color space


	//Lab color space

	// Covert color space
	Mat image_lab;
	cvtColor(img, image_lab, COLOR_BGR2Lab);
	//Split image in Lab color space (L,a,b)
	Mat lab[3];			
	split(image_lab, lab); 
	//Equalize only L channel in Lab space
	Mat EqualLabImg[3];
	EqualLabImg[1] = lab[1];
	EqualLabImg[2] = lab[2];
	equalizeHist(lab[0], EqualLabImg[0]);
	//Merge the 3 channel with L equalized
	Mat outputLabImg;
	merge(EqualLabImg, 3, outputLabImg);
	//Reconvert in BGR color space
	cvtColor(outputLabImg, finalLabImg, COLOR_Lab2BGR);
	//Split BGR equalized (in Lab) image in B G R channels
	Mat bgr_lab[3];			
	split(finalLabImg, bgr_lab);
	//Compute histograms of 3 channels
	Mat labHist[3];
	for (int i = 0; i < size(bgr); i++) {
		calcHist(&bgr_lab[i], 1, 0, Mat(), labHist[i], 1, histSize, ranges, true, false);
	}
	//show equalized image and histograms
	namedWindow("Image", WINDOW_NORMAL);
	imshow("Image", finalLabImg);
	vector<Mat> labHists = { labHist[0], labHist[1], labHist[2] };
	showHistogram(labHists);
	
	waitKey(0);
	destroyAllWindows();

	//--------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------

	//IMAGE FILTERING

	//--------------------------------------------------------------------------------------------------------


	//Gaussian

	gauss = GaussianFilter(finalLabImg, ksize, sigmaX);
		 
	namedWindow("GaussianBlur", WINDOW_NORMAL);

	createTrackbar("ksize", "GaussianBlur", &ksize, 50, on_trackbar_Gauss);
	createTrackbar("sigma", "GaussianBlur", &sigmaX, 50, on_trackbar_Gauss);

	waitKey(0);
	destroyAllWindows();


	//Median

	median = MedianFilter(finalLabImg, ksize);

	namedWindow("MedianBlur", WINDOW_NORMAL);

	createTrackbar("ksize", "MedianBlur", &ksize, 50, on_trackbar_Median);

	waitKey(0);
	destroyAllWindows();


	//Bilateral

	bilateral = BilateralFilter(finalLabImg, ksize, sigmaR, sigmaS);

	namedWindow("BilateralFilter", WINDOW_NORMAL);

	createTrackbar("sigma_range", "BilateralFilter", &sigmaR, 200, on_trackbar_Bilateral);
	createTrackbar("sigma_space", "BilateralFilter", &sigmaS, 200, on_trackbar_Bilateral);

	waitKey(0);
	destroyAllWindows();

	//--------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------

	//MORPHOLOGICAL OPERATOR

	//--------------------------------------------------------------------------------------------------------
	
	//Erode
	
	namedWindow("Erode", WINDOW_NORMAL);

	Erode(0, 0);
	createTrackbar("Kernel size:\n 2n +1", "Erode", &erosion_size, max_kernel_size, Erode);


	waitKey(0);
	destroyAllWindows();

	//Dilate

	namedWindow("Dilate", WINDOW_NORMAL);

	Dilate(0, 0);
	createTrackbar("Kernel size:\n 2n +1", "Dilate", &dilate_size, max_kernel_size, Dilate);

	waitKey(0);
	destroyAllWindows();

	//morphologyEx

	namedWindow("morphologyEx", WINDOW_NORMAL); // Create window

	Morphology_Operations(0, 0);

	createTrackbar("Operator:\n 0: Opening - 1: Closing", "morphologyEx", &morph_operator, max_operator, Morphology_Operations);
	createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", "morphologyEx", &morph_elem, max_elem, Morphology_Operations);
	createTrackbar("Kernel size:\n 2n +1", "morphologyEx", &morph_size, max_kernel_size, Morphology_Operations);


	waitKey(0);
}

void on_trackbar_Gauss(int, void*) {
	gauss.setSize(ksize);
	gauss.changeSigma(sigmaX);
	gauss.doFilter();
	imshow("GaussianBlur", gauss.getResult());
}

void on_trackbar_Median(int, void*) {
	median.setSize(ksize);
	median.doFilter();
	imshow("MedianBlur", median.getResult());
}

void on_trackbar_Bilateral(int, void*) {
	bilateral.changeSigmaRange(sigmaR);
	bilateral.changeSigmaSpace(sigmaS);
	bilateral.doFilter();
	imshow("BilateralFilter", bilateral.getResult());
}

void Morphology_Operations(int, void*) {
	// Since MORPH_X : 2,3
	int operation = morph_operator + 2;
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(finalLabImg, ImgMorphEx, operation, element);
	imshow("morphologyEx", ImgMorphEx);
}

void Erode(int, void*) {
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	erode(finalLabImg, ImgErode, element);
	imshow("Erode", ImgErode);
}

void Dilate(int, void*) {
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * dilate_size + 1, 2 * dilate_size + 1), Point(dilate_size, dilate_size));
	dilate(finalLabImg, ImgDilate, element);
	imshow("Dilate", ImgDilate);
}

// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
	// Min/Max computation
	double hmax[3] = { 0,0,0 };
	double min;
	cv::minMaxLoc(hists[0], &min, &hmax[0]);
	cv::minMaxLoc(hists[1], &min, &hmax[1]);
	cv::minMaxLoc(hists[2], &min, &hmax[2]);

	std::string wname[3] = { "blue", "green", "red" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
							 cv::Scalar(0,0,255) };

	std::vector<cv::Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++)
	{
		canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
		{
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
	}
}