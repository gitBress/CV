#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>	
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ml/ml.hpp>
#include <sstream>
#include "FindChar.h"
#include "FindPlate.h"
#include "Classifier.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

vector<vector<Point>> sortMatchingChars(vector<vector<Point>> matchingChars);

int main()
{
	bool exit = false;
	while (!exit) {

		// Loading images
		int NUM_IMG = 7;
		vector<Mat> img(NUM_IMG);

		for (int i = 0; i < img.size(); i++) {

			img[i] = imread(".\\images_T1\\" + to_string(i + 1) + ".jpg");
			if (img[i].empty()) {
				cout << "Error in image loading";
				return 2;
			}

		}
		//----------------------------------------------------------------------------
		// Select the image 

		cout << "Write a number between 0 and 6 to select an image: \n";
		int choice;
		cin >> choice;
		if (choice < 0 || choice > 6) {

			cout << "error";
			return 2;
		}
		Mat immi = img[choice];

		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		/*imshow("test", immi);*/

		//****************************************************************************

		//----------------------------------------------------------------------------
		// Preprocessing
		FindPlate PlateFinder = FindPlate::FindPlate();
		PlateFinder.preprocess(immi);

		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		/*imshow("test1", immi);
		imshow("test2", PlateFinder.imagePrep);*/

		//****************************************************************************

		//----------------------------------------------------------------------------
		// Contours extraction

		PlateFinder.getCONT(PlateFinder.imagePrep);

		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		//Mat imageCONT(PlateFinder.imagePrep.size(), CV_8UC3, Scalar(0.0, 0.0, 0.0));
		//for (int i = 0; i < PlateFinder.Contours.size(); i++) {    // for each contour

		//	drawContours(imageCONT, PlateFinder.Contours, i, Scalar(255.0, 255.0, 255.0));
		//}
		//imshow("test3", imageCONT);

		//****************************************************************************

		//----------------------------------------------------------------------------
		// Possible chars

		vector<vector<Point>> possibleCHAR;
		int countOfPossibleChars = 0;
		for (int i = 0; i < PlateFinder.Contours.size(); i++) {
			if (PlateFinder.checkIfChar(PlateFinder.Contours[i]) == true) {
				possibleCHAR.push_back(PlateFinder.Contours[i]);
				countOfPossibleChars = countOfPossibleChars + 1;
			}
		}

		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		//Mat firstPC(PlateFinder.imagePrep.size(), CV_8UC3, Scalar(0.0, 0.0, 0.0));
		//for (int i = 0; i < possibleCHAR.size(); i++) {            // for each contour
		//	Rect boundRect = boundingRect(possibleCHAR[i]);
		//	rectangle(firstPC, boundRect, Scalar(0, 255.0, 0), 1, 8, 0);
		//	drawContours(firstPC, possibleCHAR, i, Scalar(255.0, 255.0, 255.0));
		//}
		//imshow("test4", firstPC);

		//****************************************************************************

		//----------------------------------------------------------------------------
		// Char selection

		vector<vector<Point>> sortedPossibleCHAR = sortMatchingChars(possibleCHAR);

		int maxNumMatch = 0;
		vector<vector<Point>> matchCHAR_temp;
		for (int i = 0; i < countOfPossibleChars; i++) {
			PlateFinder.matchingChars(sortedPossibleCHAR[i], sortedPossibleCHAR);
			if (PlateFinder.Matches.size() >= 4) {
				if (matchCHAR_temp.empty()) {		// if first set of matching chars
					matchCHAR_temp = PlateFinder.Matches;
					maxNumMatch = matchCHAR_temp.size();
				}
				else if (PlateFinder.Matches.size() > maxNumMatch) {		// if not first set, check if it is a bigger set than the already selected one
					matchCHAR_temp = PlateFinder.Matches;					// if not bigger, discard it
					maxNumMatch = matchCHAR_temp.size();
				}
			}
		}
		if (matchCHAR_temp.empty()) {
			cout << "No matching chars found";
			return 2;
		}

		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		//Mat matchPC(PlateFinder.imagePrep.size(), CV_8UC3, Scalar(0.0, 0.0, 0.0));
		//for (int j = 0; j < matchCHAR_temp.size(); j++) {            // for each contour
		//	Rect boundRect = boundingRect(matchCHAR_temp[j]);
		//	rectangle(matchPC, boundRect, Scalar(0, 255.0, 0), 1, 8, 0);
		//	drawContours(matchPC, matchCHAR_temp, j, Scalar(255.0, 255.0, 255.0));
		//}
		//imshow("test5", matchPC);

		//****************************************************************************

		//----------------------------------------------------------------------------
		// Plate extraction 

		Rect foundPlate = PlateFinder.platePos(matchCHAR_temp);

		Mat plateGreen = immi.clone();
		Point Pt1 = Point(foundPlate.x - foundPlate.width / 2, foundPlate.y - foundPlate.height / 2);
		Point Pt2 = Point(foundPlate.x + foundPlate.width / 2, foundPlate.y + foundPlate.height / 2);
		rectangle(plateGreen, Pt1, Pt2, Scalar(78.0, 123.0, 216.0), 2, 8, 0);
		namedWindow("plate", WINDOW_NORMAL);
		imshow("plate", plateGreen);

		Mat imgPlate(Size(foundPlate.width, foundPlate.height), CV_8UC3, Scalar(0.0, 0.0, 0.0));
		imgPlate = PlateFinder.extractPlateImg(immi, foundPlate);

		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		/*imshow("test6", imgPlate);*/

		//****************************************************************************

		//----------------------------------------------------------------------------
		// Plate preprocessing

		FindChar CharToFind = FindChar::FindChar(imgPlate);
		CharToFind.preprocCHAR();
		Mat prepPlate = CharToFind.preprocPlate;
		
		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		/*imshow("test7", prepPlate);*/

		//****************************************************************************

		CharToFind.contPlate();
		Mat PlateCONT(imgPlate.size(), CV_8UC3, Scalar(0.0, 0.0, 0.0));

		Mat CharCONT(imgPlate.size(), CV_8UC3, Scalar(0.0, 0.0, 0.0));
		vector<vector<Point>> chars;
		for (int j = 0; j < CharToFind.contours.size(); j++) {
			if (CharToFind.maybeChar(CharToFind.contours[j]) == true) {
				chars.push_back(CharToFind.contours[j]);
			}
		}

		//****************************************************************************
		// Uncomment below to show images of intermediate steps

		/*for (int i = 0; i < chars.size(); i++) {

			Rect boundRect = boundingRect(chars[i]);
			rectangle(CharCONT, boundRect, Scalar(0, 255.0, 0), 1, 8, 0);
			drawContours(CharCONT, chars, i, Scalar(255.0, 255.0, 255.0));
		}
		imshow("test8", CharCONT);*/

		//****************************************************************************

		// Chars sorting
		chars = sortMatchingChars(chars);

		vector<Mat> charsInPlate;
		for (int i = 0; i < chars.size(); i++) {
			Mat char_temp = CharToFind.extractChar(prepPlate, chars[i]);
			charsInPlate.push_back(~char_temp);

		}

		//----------------------------------------------------------------------------
		// Character recognition

		int number_of_class = 35;
		int number_of_sample = 12;

		vector<Mat> trainingImages;
		for (int i = 0; i < number_of_class; i++) {

			for (int j = 0; j < number_of_sample; j++) {

				Mat trainingImages_tmp = imread(".\\data\\" + to_string(i) + "\\" + to_string(j + 1) + ".jpg");
				if (trainingImages_tmp.empty()) {
					cout << "Error in image loading";
					return 2;
				}

				// Processing to have all equal images
				cvtColor(trainingImages_tmp, trainingImages_tmp, COLOR_BGR2GRAY);
				resize(trainingImages_tmp, trainingImages_tmp, Size(20, 30), 0, 0, INTER_LANCZOS4);

				trainingImages.push_back(trainingImages_tmp);
			}
		}
		vector<float> mapping = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 78, 80, 83, 84, 85, 86, 88, 89, 90, 82, 73, 81, 87, 74 };


		Classifier KNNclassifier = Classifier::Classifier();
		string strPlate = KNNclassifier.trainAndClassify(trainingImages, mapping, number_of_class, number_of_sample, charsInPlate);

		cout << "Plate read = " << strPlate << "\n";

		waitKey(0);
		destroyAllWindows();

		cout << "Do you want to exit? y/n \n";
		string end;
		cin >> end;
		if (end != "y" && end != "n") {

			cout << "error";
			return 2;
		}
		else {
			if (end == "y") {
				exit = true;
			}
		}
	}
		return 0;
	
}

	vector<vector<Point>> sortMatchingChars(vector<vector<Point>> matchingChars) {

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

