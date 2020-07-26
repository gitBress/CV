#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>	
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "PanoramicImage.h"
#include "panoramic_utils.h"

using namespace cv;
using namespace std;

int main()
{
	//Loading images
	int NUM_IMG = 13;
	vector<Mat> img(NUM_IMG);

	for (int i = 0; i < img.size(); i++) {

		img[i] = imread(".\\data_lab5\\i0" + to_string(i + 1) + ".png");
		if (img[i].size == 0) {
			cout << "Error in image loading";
			return 2;
		}
		/*imshow("prova" + to_string(i + 1), img[i]);*/
	}

	//Create the object
	PanoramicImage panoramic = PanoramicImage(img, NUM_IMG);


	//project image in cylindrical surface
	double angle = 33; 
	vector<Mat> projected;

	projected = panoramic.ProjectImages(angle);
	

	//extract keypoints (and descriptors)
	vector<vector<KeyPoint>> kpts = panoramic.ExtractORB();

	//matches
	vector<vector<DMatch>> match = panoramic.Matcher();

	//refine matches
	int ratio = 5;

	vector<vector<DMatch>> refined = panoramic.RefinedMatcher(ratio);

	//show refined matches
	/*vector<Mat> outimg(NUM_IMG-1);
	for (int i = 0; i < NUM_IMG - 1; i++) {
		drawMatches(projected[i], kpts[i], projected[i+1], kpts[i+1], refined[i], outimg[i]);
		namedWindow("prova" + to_string(i + 1), WINDOW_NORMAL);
		imshow("prova" + to_string(i + 1), outimg[i]);
		waitKey(0);
	}*/

	vector<vector<Point2f>> good_Match = panoramic.findKeyPointsHomography();
	
	
	vector<int> deltax = panoramic.find_dX();
	for (int i = 0; i < deltax.size(); i++) {
		cout << "delta x : " << deltax[i] << "\n";
	}


 	Mat stitched = panoramic.Stitching();
	namedWindow("Panoramic", WINDOW_NORMAL);
	imshow("Panoramic", stitched);


	waitKey(0);
	return 0;
}

