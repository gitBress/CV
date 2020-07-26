#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "PanoramicImage.h"
#include "panoramic_utils.h"

using namespace cv;

	// constructor
	PanoramicImage::PanoramicImage(vector<Mat> img_set, int n_img) {

		image_set = img_set;
		n_image = n_img;
	}
	
	// project the images on a cylinder surface
	vector<Mat> PanoramicImage::ProjectImages(double angle) {
		
		vector<Mat> cylindrical(n_image);
		for (int i = 0; i < n_image; i++) {
			cylindrical[i] = PanoramicUtils::cylindricalProj(image_set[i], angle);
		}
		resultProjection = cylindrical;
		return resultProjection;
	}

	// extract ORB features from the images
	vector<vector<KeyPoint>> PanoramicImage::ExtractORB() {

		// Initiate ORB detector
		Ptr<ORB> detector = ORB::create(50000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

		vector<vector<KeyPoint>> kp(n_image);
		vector<Mat> descr(n_image);

		for (int i = 0; i < n_image; i++) {
			
			// find the keypoints and descriptors with ORB
			detector->detectAndCompute(resultProjection[i], Mat(), kp[i], descr[i]);
			
		}

		KeyPoints = kp;
		Descriptors = descr;

		return KeyPoints;
	}

	// find matches
	vector<vector<DMatch>> PanoramicImage::Matcher() {

		// Initiate matcher
		BFMatcher desc_matcher(NORM_HAMMING, false);

		vector<vector<DMatch>> mtch(n_image - 1);

		for (int i = 0; i < n_image - 1; i++) {
			desc_matcher.match(Descriptors[i], Descriptors[i+1], mtch[i], Mat());
		}

		Matches = mtch;
		return Matches;
	}

	// refines matches
	vector<vector<DMatch>> PanoramicImage::RefinedMatcher(int ratio) {
		
		// max and min distances between keypoints

		vector<double> min_dist(n_image - 1);

		for (int i = 0; i < n_image - 1; i++) {

			vector<DMatch> temp_m = Matches[i];
			double mini = 1000;

			for (int j = 0; j < temp_m.size(); j++){

				double dist = temp_m[j].distance;
				if (dist < mini) mini = dist;
			}
			printf("-- Min dist : %f \n", mini);
			min_dist[i] = mini;
		}

		//Refine the matches found above by selecting the matches with distance less than ratio * min_distance

		vector<vector<DMatch>> good_matches;

		for (int i = 0; i < n_image - 1; i++) {

			
			vector<DMatch> temp_matches = Matches[i];
			vector<DMatch> temp_good;
			int temp_min = min_dist[i];
			if (temp_min == 0) {
				temp_min = temp_min + 1;
			}
			for (int j = 0; j < temp_matches.size(); j++) {

				if (temp_matches[j].distance < ratio * temp_min) {

					temp_good.push_back(temp_matches[j]);
				}
			}
			good_matches.push_back(temp_good);
			
		}

		RefinedMatches = good_matches;
		return RefinedMatches;
	}

	vector<vector<Point2f>> PanoramicImage::findKeyPointsHomography() {

		vector<vector<Point2f>> good_query(n_image - 1);
		vector<vector<Point2f>> good_train(n_image - 1);
		vector<vector<uchar>> temp_mask(n_image - 1);

		for (int i = 0; i < n_image - 1; i++) {

			vector<uchar> temp_M;
			vector<DMatch> temp_refMatch = RefinedMatches[i];
			vector<KeyPoint> temp_kpts1 = KeyPoints[i];
			vector<KeyPoint> temp_kpts2 = KeyPoints[i+1];

			vector<Point2f> pts1;
			vector<Point2f> pts2;
			for (int j = 0; j < temp_refMatch.size(); j++) {
				pts1.push_back(temp_kpts1[temp_refMatch[j].queryIdx].pt);
				pts2.push_back(temp_kpts2[temp_refMatch[j].trainIdx].pt);
			}

			findHomography(pts1, pts2, RANSAC, 3, temp_M);

			vector<Point2f> temp_good_query;
			for (int k = 0; k < temp_M.size(); k++) {

				if (static_cast<int>(temp_M[k]) == 1) {

					temp_good_query.push_back(pts1[k]);

				}
			}
			good_query[i] = temp_good_query;


			vector<Point2f> temp_good_train;
			for (int k = 0; k < temp_M.size(); k++) {

				if (static_cast<int>(temp_M[k]) == 1) {

					temp_good_train.push_back(pts2[k]);

				}
			}
			good_train[i] = temp_good_train;

			temp_mask[i] = temp_M;

			
		}
		goodMatches_query = good_query;
		goodMatches_train = good_train;
		MatchMask = temp_mask;
	
		return goodMatches_query;
	}
	
	vector<int> PanoramicImage::find_dX() {
		
		vector<int> meanDX_Q;
		vector<int> meanDX_T;
		
		for (int i = 0; i < n_image - 1; i++) {
			
			vector<Point2f> temp_GM_query = goodMatches_query[i];
			
			double temp_Mdx_Q = 0;
			for (int j = 0; j < temp_GM_query.size(); j++) {
				temp_Mdx_Q = temp_Mdx_Q + temp_GM_query[j].x;
		
			}
			temp_Mdx_Q = temp_Mdx_Q / temp_GM_query.size();

			vector<Point2f> temp_GM_train = goodMatches_train[i];

			double temp_Mdx_T = 0;
			for (int j = 0; j < temp_GM_train.size(); j++) {
				temp_Mdx_T = temp_Mdx_T + temp_GM_train[j].x;

			}
			temp_Mdx_T = temp_Mdx_T / temp_GM_train.size();


			meanDX_Q.push_back(temp_Mdx_Q);
			meanDX_T.push_back(temp_Mdx_T);
		
			
		}
		

		deltaX_Q = meanDX_Q;
		deltaX_T = meanDX_T;
		
		return deltaX_Q;
	}

	Mat PanoramicImage::Stitching() {

		// find dimension of the matrix
		vector<int> sum_QT;
		
		
		int dim_x = 0;
		for (int j = 0; j < deltaX_Q.size(); j++) {
			sum_QT.push_back(dim_x + deltaX_Q[j]);
			dim_x = dim_x + deltaX_Q[j] - deltaX_T[j];
			
		}
		dim_x = dim_x + resultProjection[n_image-1].cols;

		int dim_y = resultProjection[n_image - 1].rows;

		Mat pan_mat(dim_y, dim_x, CV_8UC1, Scalar(0));

		resultProjection[0](Rect(0, 0, deltaX_Q[0], resultProjection[0].rows)).copyTo(pan_mat(Rect(0, 0, deltaX_Q[0], resultProjection[0].rows)));
		
		for (int i = 0; i < n_image-2; i++) {
			Mat temp_img = resultProjection[i+1];
			
			temp_img(Rect(deltaX_T[i], 0, temp_img.cols - deltaX_T[i], temp_img.rows)).copyTo(pan_mat(Rect(sum_QT[i], 0, temp_img.cols - deltaX_T[i], temp_img.rows)));
	
			
		}
		
		int col = resultProjection[n_image - 1].cols - deltaX_T[n_image - 2];
		int row = resultProjection[n_image - 1].rows;
		resultProjection[n_image - 1](Rect(deltaX_T[n_image - 2], 0, col, row)).copyTo(pan_mat(Rect(sum_QT[n_image - 2], 0, col, row)));

		
		Panorama = pan_mat;

		return Panorama;
	}
