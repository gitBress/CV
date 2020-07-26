#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "panoramic_utils.h"

using namespace cv;
using namespace std;


class PanoramicImage {

// Methods

public:

	// constructor 
	PanoramicImage(vector<Mat> img_set, int n_img);

	// project the images on a cylinder surface
	vector<Mat> ProjectImages(double angle);

	// extract ORB features from the images
	vector<vector<KeyPoint>> ExtractORB();

	// find matches
	vector<vector<DMatch>> Matcher();

	// refines matches
	vector<vector<DMatch>> RefinedMatcher(int ratio);

	// finds key points homography
	vector<vector<Point2f>> findKeyPointsHomography();

	vector<int> find_dX();

	// merge images
	Mat Stitching();


// Data

protected:

	// set of images
	vector<Mat> image_set;
	
	//number of images
	int n_image;

	//angle
	int angle;

	// projected images on cylinder surface
	vector<Mat> resultProjection;

	// keypoints and descriptors extracted with orb
	vector<vector<KeyPoint>> KeyPoints;
	vector<Mat> Descriptors;

	// matches
	vector<vector<DMatch>> Matches;

	// refined matches
	vector<vector<DMatch>> RefinedMatches;

	// mask
	vector<vector<uchar>> MatchMask;
	vector<vector<Point2f>> goodMatches_query;
	vector<vector<Point2f>> goodMatches_train;
	vector<Mat> H;

	//find delta x
	vector<int> deltaX_Q;
	vector<int> deltaX_T;

	// stitched
	Mat Panorama;
};