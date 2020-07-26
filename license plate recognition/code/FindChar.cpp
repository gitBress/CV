#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include "FindChar.h"


using namespace cv;
using namespace std;

//------------------------------------------------------------------------------------------------------
// PREPROCESSING OF THE PLATE
//------------------------------------------------------------------------------------------------------

FindChar::FindChar(Mat plate) {
	origPlate = plate;
}

void FindChar::preprocCHAR() {

	// Greyscale
	Mat greyPlate;
	cvtColor(origPlate, greyPlate, CV_BGR2GRAY);

	// Denoising
	Mat blurredPlate;
	float h = 30.0;
	int templateWindowSize = 7;
	int searchWindowSize = 21;
	fastNlMeansDenoising(greyPlate, blurredPlate, h, templateWindowSize, searchWindowSize);

	// Thresholding
	Mat threshPlate;
	int ADAPTIVE_THRESH_BLOCK_SIZE = 49;
	int ADAPTIVE_THRESH_WEIGHT = 1;
	double maxValue = 255.0;
	adaptiveThreshold(blurredPlate, threshPlate, maxValue, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);

	// Erode
	Point anchor = Point(-1, -1);
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 6), anchor);
	erode(threshPlate, preprocPlate, element);
}

void FindChar::contPlate() {
	
	findContours(preprocPlate, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

}

bool FindChar::maybeChar(vector<Point> cont) {
	vector<Point> possibleChar = cont;
	Rect boundRect = boundingRect(possibleChar);
	float boundingRectArea = boundRect.width * boundRect.height;
	float aspectRatio = float(boundRect.width / float(boundRect.height));

	if (boundingRectArea > 2000 && boundingRectArea < 12000 && boundRect.height < 240
		&& boundRect.height > 150 && aspectRatio > 0.001 && aspectRatio < 50) {
		return true;
	}
	else {
		return false;
	}
}

Mat FindChar::extractChar(Mat origImg, vector<Point> oneChar) {
	Mat imagePlate = origImg.clone();
	Rect Rectangle = boundingRect(oneChar);
	float centerX = (Rectangle.x + Rectangle.x + Rectangle.width) / 2;
	float centerY = (Rectangle.y + Rectangle.y + Rectangle.height) / 2;
	Size charSize = Size(Rectangle.width + 1, Rectangle.height + 3);
	Mat imgChar(charSize, CV_8UC3, Vec3b(0.0, 0.0, 0.0));
	Point2f charCenter;
	charCenter.x = centerX;
	charCenter.y = centerY;

	int patchType = -1;
	getRectSubPix(imagePlate, charSize, charCenter, imgChar, patchType);

	return imgChar;
}