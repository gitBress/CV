#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>
#include "FindPlate.h"


using namespace cv;
using namespace std;

FindPlate::FindPlate(){}

//------------------------------------------------------------------------------------------------------
// PREPROCESSING OF THE IMAGE
//------------------------------------------------------------------------------------------------------

void FindPlate::preprocess(Mat imgOriginal) {

	Mat image = imgOriginal;

	// Convert to greyscale
	Mat imageGREY;
	cvtColor(image, imageGREY, cv::COLOR_BGR2GRAY);  

	// Morph to maximize contrast
	Mat imageMORPH = Morph(imageGREY);

	// Smoothing
	Mat imageBLURRED;
	Size GAUSSIAN_SMOOTH_FILTER_SIZE = Size(5, 5);
	double sigmaX = 0;
	GaussianBlur(imageMORPH, imageBLURRED, GAUSSIAN_SMOOTH_FILTER_SIZE, sigmaX);
	
	//// Thresholding
	int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
	int ADAPTIVE_THRESH_WEIGHT = 9;
	double maxValue = 255.0;
	Mat imageTHRESH;
	adaptiveThreshold(imageBLURRED, imageTHRESH, maxValue, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);

	imagePrep = imageTHRESH;
}

//------------------------------------------------------------------------------------------------------
// METHODS FOR PREPROCESSING
//------------------------------------------------------------------------------------------------------

Mat FindPlate::Morph(Mat greyImage) {

	Mat IMGgrey = greyImage;

	Mat IMGtophat;
	Mat IMGblackhat;
	Mat imgGrayscalePlusTopHat;
	Mat imgGrayscalePlusTopHatMinusBlackHat;

	Mat structuringElement = getStructuringElement(CV_SHAPE_RECT, Size(3, 3));

	morphologyEx(IMGgrey, IMGtophat, CV_MOP_TOPHAT, structuringElement);
	morphologyEx(IMGgrey, IMGblackhat, CV_MOP_BLACKHAT, structuringElement);

	imgGrayscalePlusTopHat = IMGgrey + IMGtophat;
	imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - IMGblackhat;

	return(imgGrayscalePlusTopHatMinusBlackHat);
}

//------------------------------------------------------------------------------------------------------
// GETTING CONTOURS
//------------------------------------------------------------------------------------------------------

void FindPlate::getCONT(Mat imgTHR) {

	Mat imageTHRESH = imgTHR;
	findContours(imageTHRESH, Contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
}

//------------------------------------------------------------------------------------------------------
// SEARCH FOR CHAR
//------------------------------------------------------------------------------------------------------

bool FindPlate::checkIfChar(vector<Point> cont) {
	vector<Point> possibleChar = cont;
	Rect boundRect = boundingRect(possibleChar);
float boundingRectArea = boundRect.width * boundRect.height;
float aspectRatio = float(boundRect.width / float(boundRect.height));

if (boundingRectArea > 25 && boundingRectArea < 750 && boundRect.width > 1
	&& boundRect.height > 3 && 0.7 < aspectRatio < 1) {
	return true;
}
else {
	return false;
}
}

float FindPlate::distanceBetweenChars(vector<Point> firstChar, vector<Point> secondChar) {
	Rect boundRect1 = boundingRect(firstChar);
	float centerX1 = (boundRect1.x + boundRect1.x + boundRect1.width) / 2;
	float centerY1 = (boundRect1.y + boundRect1.y + boundRect1.height) / 2;
	Rect boundRect2 = boundingRect(secondChar);
	float centerX2 = (boundRect2.x + boundRect2.x + boundRect2.width) / 2;
	float centerY2 = (boundRect2.y + boundRect2.y + boundRect2.height) / 2;

	float x = abs(centerX1 - centerX2);
	float y = abs(centerY1 - centerY2);

	return sqrt((x * 2) + (y * 2));
}

float FindPlate::angleBetweenChars(vector<Point> firstChar, vector<Point> secondChar) {
	Rect boundRect1 = boundingRect(firstChar);
	float centerX1 = (boundRect1.x + boundRect1.x + boundRect1.width) / 2;
	float centerY1 = (boundRect1.y + boundRect1.y + boundRect1.height) / 2;
	Rect boundRect2 = boundingRect(secondChar);
	float centerX2 = (boundRect2.x + boundRect2.x + boundRect2.width) / 2;
	float centerY2 = (boundRect2.y + boundRect2.y + boundRect2.height) / 2;

	float adjacent = float(abs(centerX1 - centerX2));
	float opposite = float(abs(centerY1 - centerY2));

	float angleInRad;
	if (adjacent != 0.0) {
		angleInRad = atan(opposite / adjacent);
	}
	else {
		angleInRad = 1.5708;
	}

	float angleInDeg = angleInRad * (180.0 / CV_PI);

	return angleInDeg;

}

float FindPlate::getDiagSize(vector<Point> foundChar) {
	Rect boundRect = boundingRect(foundChar);
	float diagonalSize = sqrt((boundRect.width * 2) + (boundRect.height * 2));

	return diagonalSize;
}

void FindPlate::matchingChars(vector<Point> possibleC, vector<vector<Point>> possibleChars) {

	vector<vector<Point>> matchingCHAR; 
	matchingCHAR.push_back(possibleC);

	float distance;
	float angle;
	float changeInArea;
	float changeInWidth;
	float changeInHeight;
	Rect boundRect1 = boundingRect(possibleC);
	float boundRect1Area = boundRect1.width * boundRect1.height;
	float diagSize1 = getDiagSize(possibleC);

	for (int i = 0; i < possibleChars.size(); i++) {
		if (possibleChars[i] != possibleC) {

			// Computation
			distance = distanceBetweenChars(possibleC, possibleChars[i]);
			angle = angleBetweenChars(possibleC, possibleChars[i]);

			Rect boundRect2 = boundingRect(possibleChars[i]);
			float boundRect2Area = boundRect2.width * boundRect2.height;
			changeInArea = float(abs(boundRect2Area - boundRect1Area)) / float(boundRect1Area);

			changeInWidth = float(abs(boundRect2.width - boundRect1.width)) / float(boundRect1.width);

			changeInHeight = float(abs(boundRect2.height - boundRect1.height)) / float(boundRect1.height);

			// Check
			if (distance < (diagSize1 * 3) && angle < 5.0 && changeInArea < 0.5 && changeInWidth < 0.8 && changeInHeight < 0.2) {
				matchingCHAR.push_back(possibleChars[i]);
			}
		}
	}
	Matches = matchingCHAR;
}

vector<vector<Point>> FindPlate::sortMatchingChars(vector<vector<Point>> matchingChars) {
	
	struct Left_Right_contour_sorter 
	{
		bool operator ()(const vector<Point>& a, const vector<Point> & b)
		{
			Rect ra(boundingRect(a));
			Rect rb(boundingRect(b));
			return (ra.x < rb.x);
		}
	};
	
	vector<vector<Point>> sortedMC = matchingChars;
	sort(sortedMC.begin(), sortedMC.end(), Left_Right_contour_sorter());

 	return sortedMC;
}

//------------------------------------------------------------------------------------------------------
// DISPLAY POSSIBLE PLATE
//------------------------------------------------------------------------------------------------------

Rect FindPlate::platePos(vector<vector<Point>> matchingChars) {
	vector<vector<Point>> sortedMC = sortMatchingChars(matchingChars);
	Rect plate;

	// center of the plate
	Rect boundRect1 = boundingRect(sortedMC[0]); //center of first char
	float centerX1 = (boundRect1.x + boundRect1.x + boundRect1.width) / 2;
	float centerY1 = (boundRect1.y + boundRect1.y + boundRect1.height) / 2;
	Rect boundRect2 = boundingRect(sortedMC[sortedMC.size() - 1]); //center of last char
	float centerX2 = (boundRect2.x + boundRect2.x + boundRect2.width) / 2;
	float centerY2 = (boundRect2.y + boundRect2.y + boundRect2.height) / 2;
	plate.x = (centerX1 + centerX2) / 2.0;
	plate.y = (centerY1 + centerY2) / 2.0;

	// plate width and height
	plate.width = int((centerX2 + boundRect2.width - centerX1) * 1.1);

	int totalH = 0;
	for (int i = 0; i < sortedMC.size(); i++) {
		Rect boundRect = boundingRect(sortedMC[i]);
		totalH = totalH + boundRect.height;
	}
	float avgH = totalH / sortedMC.size();
	plate.height = int(avgH * 1.5);

	return plate;
}

Mat FindPlate::extractPlateImg(Mat imgRotated, Rect plate) {
	Mat imgPlate(Size(plate.width, plate.height), CV_8UC3, Vec3b(0.0, 0.0, 0.0));
	Point2f plateCenter;
	plateCenter.x = plate.x;
	plateCenter.y = plate.y;
	getRectSubPix(imgRotated, Size(plate.width, plate.height), plateCenter, imgPlate);
	
	Mat imgPlateInterpol(Size(500, 250), CV_8UC3, Vec3b(0.0, 0.0, 0.0));
	double fx = 0;
	double fy = 0;
	resize(imgPlate, imgPlateInterpol, imgPlateInterpol.size(), fx, fy, INTER_LANCZOS4);

	return imgPlateInterpol;
}

