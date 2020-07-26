#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>



class FindPlate {
public:
	FindPlate();

	void preprocess(cv::Mat imgOriginal);
	cv::Mat Morph(cv::Mat image);
	void getCONT(cv::Mat imgTHR);
	bool checkIfChar(std::vector<cv::Point> cont);
	float distanceBetweenChars(std::vector<cv::Point> firstChar, std::vector<cv::Point> secondChar);
	float angleBetweenChars(std::vector<cv::Point> firstChar, std::vector<cv::Point> secondChar);
	float getDiagSize(std::vector<cv::Point> foundChar);
	void matchingChars(std::vector<cv::Point> possibleC, std::vector<std::vector<cv::Point>> possibleChars);
	std::vector<std::vector<cv::Point>> sortMatchingChars(std::vector<std::vector<cv::Point>> matchingChars);
	cv::Rect platePos(std::vector<std::vector<cv::Point>> matchingChars);
	cv::Mat extractPlateImg(cv::Mat imgRotated, cv::Rect plate);


	cv::Mat imagePrep;
	std::vector<std::vector<cv::Point>> Contours;
	std::vector<std::vector<cv::Point>> Matches;

};