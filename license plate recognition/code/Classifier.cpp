#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "Classifier.h"


using namespace cv;
using namespace std;
using namespace cv::ml;


Classifier::Classifier(){}

string Classifier::trainAndClassify(vector<Mat> Images, vector<float> Map, int Classes, int Samples, vector<Mat> chars) {

	int number_of_class = Classes;
	int number_of_sample = Samples;
	vector<Mat> trainingImages = Images;
	vector<float> mapping = Map;

	//------------------------------------------------------------------------------------------------------
	// RESHAPE IMAGES AND LABELS
	//------------------------------------------------------------------------------------------------------

	Mat trainImgFlattenedFloat(Size(600, number_of_class*number_of_sample), CV_32FC1);
	for (int i = 0; i < trainingImages.size(); i++) {

		// Multiple images as vectors into a single Mat
		Mat trainingImagesFloat;
		trainingImages[i].convertTo(trainingImagesFloat, CV_32FC1);
		Mat trainingImagesReshapedFlattenedFloat = trainingImagesFloat.reshape(1, 1);

		for (int j = 0; j < 600; j++) {
			trainImgFlattenedFloat.at<float>(i, j) = trainingImagesReshapedFlattenedFloat.at<float>(0, j);
		}
	}

	Mat trainingLabelsFlattenedFloat(Size(1, number_of_class*number_of_sample), CV_32FC1);
	int count = 0;
	for (int i = 0; i < number_of_class; i++) {

		for (int j = 0; j < number_of_sample; j++) {

			// Labels as one single vector in a Mat 
			trainingLabelsFlattenedFloat.at<float>(count, 0) = mapping[i];
			count++;
		}
	}

	//------------------------------------------------------------------------------------------------------
	// ISTANTIATE AND TRAIN KNN ALGORITHM
	//------------------------------------------------------------------------------------------------------

	Ptr<ml::KNearest>  kNearest(ml::KNearest::create());
	kNearest->train(trainImgFlattenedFloat, ml::ROW_SAMPLE, trainingLabelsFlattenedFloat);

	//------------------------------------------------------------------------------------------------------
	// CHAR CLASSIFICATION
	//------------------------------------------------------------------------------------------------------

	string strPlate;
	int resized_width = 20;
	int resized_height = 30;

	for (int i = 0; i < chars.size(); i++) {

		// Resize and reshape image 
		Mat charsInPlateResized;
		double fx = 0;
		double fy = 0;
		resize(chars[i], charsInPlateResized, Size(resized_width, resized_height), fx, fy, INTER_LANCZOS4);

		Mat charsInPlateFloat;
		charsInPlateResized.convertTo(charsInPlateFloat, CV_32FC1);             
		Mat charsInPlateFlattenedFloat = charsInPlateFloat.reshape(1, 1);

		// Call KNN
		Mat matCurrentChar(0, 0, CV_32F);
		int k = 1;
		kNearest->findNearest(charsInPlateFlattenedFloat, k, matCurrentChar);

		// Concatenate string
		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);
		strPlate = strPlate + char(int(fltCurrentChar));
	}
	return strPlate;
}
