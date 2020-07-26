#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>	
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat img;
Mat image;

void hough_lines(int, void*);
Mat edges;
Mat cedges;
Mat out;
Mat edges_line;
int rho;
int theta;
int thresh;
int min_thresh = 50; 
vector<Vec2f> lines;

void hough_circles(int, void*);
Mat ccircles;
Mat grey_circles;
int minDist = 100;
int param1 = 2;
int param2 = 20;
int minRadius = 10;
int maxRadius = 20;
vector<Vec3f> circles;

int main()
{

	//--------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------

	//IMAGE LOADING

	//--------------------------------------------------------------------------------------------------------


    image = imread(".\\images\\road2.png");

	//Mat img; //grey image

	cvtColor(image, img, COLOR_BGR2GRAY);

	namedWindow("Image", WINDOW_NORMAL);
	imshow("Image", image);
	waitKey(0);

	/*imshow("Image", img);
	waitKey(0);*/

	//--------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------

	//LINE EXTRACTION

	//--------------------------------------------------------------------------------------------------------


	//Edge map with canny algorithm

	double thr1 = 250; //250
	double thr2 = 750; //750
	int apertureSize = 3;

	Canny(img, edges, thr1, thr2, apertureSize);
	

	imshow("Image", edges);

	waitKey(0);

	//Standard hough transform

	
	namedWindow("Hough_Lines", WINDOW_NORMAL);

	//run to found parameters:
	//----------------------
	/*createTrackbar("Rho", "Hough_Lines", &rho, 10, hough_lines);
	createTrackbar("Theta", "Hough_Lines", &theta, 10, hough_lines);
	createTrackbar("Threshold", "Hough_Lines", &thresh, 100, hough_lines);

	hough_lines(0, 0);*/
	//----------------------

	// BEST PARAMETERS FOUND: rho=2, theta=3, threshold=35

	//once parameters are found run the code:
	//----------------------
	rho = 2;
	theta = 3;
	thresh = 35;
	edges_line = edges.clone();
	cvtColor(edges_line, cedges, COLOR_GRAY2BGR);
	HoughLines(edges_line, lines, CV_PI / (1 + rho), CV_PI / (1 + theta), min_thresh + thresh, 0, 0);
	for (size_t i = 0; i < lines.size(); i++)
	{
		float phase = lines[i][0], mod = lines[i][1];
		Point pt1, pt2;
		double a = cos(mod), b = sin(mod);
		double x0 = a * phase, y0 = b * phase;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cedges, pt1, pt2, Scalar(0, 0, 255), 3);
	}
	imshow("Hough_Lines", cedges);
	//----------------------

	waitKey(0); //to go on with the selected parameters
	//destroyAllWindows();

	Mat out_lines = cedges.clone();
	namedWindow("Lines", WINDOW_NORMAL);
	/*imshow("Lines", out_lines);
	waitKey(0);*/

	//find points through which lines are passing
	vector<Point> pt1(lines.size()), pt2(lines.size());
	for (size_t i = 0; i < lines.size(); i++)
	{
		float phase = lines[i][0], mod = lines[i][1];
		double a = cos(mod), b = sin(mod);
		double x0 = a * phase, y0 = b * phase;
		pt1[i].x = cvRound(x0 + 1000 * (-b));
		pt1[i].y = cvRound(y0 + 1000 * (a));
		pt2[i].x = cvRound(x0 - 1000 * (-b));
		pt2[i].y = cvRound(y0 - 1000 * (a));
	}

	//find interception point for the two lines
	Point x = pt1[1] - pt1[0];
	Point d1 = pt2[0] - pt1[0];
	Point d2 = pt2[1] - pt1[1];

	float cross = d1.x*d2.y - d1.y*d2.x;
	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	Point r = pt1[0] + d1 * t1;
	
	vector<Point> points;
	points.push_back(pt1[0]);
	points.push_back(pt2[1]);
	points.push_back(r);
	Mat poly_img = image.clone();
	fillConvexPoly(poly_img, points, Scalar(0, 0, 255));
	imshow("Lines", poly_img);
	waitKey(0);
	/*destroyAllWindows();*/

	//--------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------

	//CIRCLE EXTRACTION

	//--------------------------------------------------------------------------------------------------------
	
	namedWindow("Hough_Circles", WINDOW_NORMAL);

	//run to found parameters:
	//----------------------
	/*createTrackbar("minDist", "Hough_Circles", &minDist, 150, hough_circles);
	createTrackbar("param1", "Hough_Circles", &param1, 30, hough_circles);
	createTrackbar("param2", "Hough_Circles", &param2, 30, hough_circles);
	createTrackbar("minRadius", "Hough_Circles", &minRadius, 30, hough_circles);
	createTrackbar("maxRadius", "Hough_Circles", &maxRadius, 30, hough_circles);

	hough_circles(0, 0);*/
	//----------------------

	// BEST PARAMETERS FOUND: mindist=100, param1=1, param2=21, minRadius=6, maxRadius=13

	//once parameters are found run the code:
	//----------------------
	minDist = 100; 
	param1 = 1; 
	param2 = 21; 
	minRadius = 6; 
	maxRadius = 13;
	grey_circles = img.clone();
	ccircles = image.clone();
	HoughCircles(grey_circles, circles, HOUGH_GRADIENT, 1, grey_circles.rows / (1 + minDist),
		1 + param1, 1 + param2, minRadius, maxRadius);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(ccircles, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(ccircles, center, radius, Scalar(0, 255, 0), -1, 8, 0);
	}
	imshow("Hough_Circles", ccircles);
	//----------------------

	waitKey(0); //to go on with the selected parameters
	/*destroyAllWindows();*/

	Mat out_circles = ccircles.clone();
	/*namedWindow("Circles", WINDOW_NORMAL);
	imshow("Circles", out_circles);
	waitKey(0);*/

	//--------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------

	//SHOW FINAL RESULT

	//--------------------------------------------------------------------------------------------------------

	Mat final_image = out_circles.clone();
	fillConvexPoly(final_image, points, Scalar(0, 0, 255));
	namedWindow("Final Result", WINDOW_NORMAL);
	imshow("Final Result", final_image);
	
	waitKey(0);
}

void hough_lines(int, void*) {
	edges_line = edges.clone();
	cvtColor(edges_line, cedges, COLOR_GRAY2BGR);
	HoughLines(edges_line, lines, CV_PI/(1 + rho) , CV_PI / (1 + theta), min_thresh + thresh, 0, 0);
	//to show the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float phase = lines[i][0], mod = lines[i][1];
		Point pt1, pt2;
		double a = cos(mod), b = sin(mod);
		double x0 = a * phase, y0 = b * phase;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cedges, pt1, pt2, Scalar(0, 0, 255), 3);
	}
	imshow("Hough_Lines", cedges);
}

void hough_circles(int, void*) {
	grey_circles = img.clone();
	ccircles = image.clone();
	HoughCircles(grey_circles, circles, HOUGH_GRADIENT, 1, grey_circles.rows / (1+minDist),
		1+param1, 1+param2, minRadius, maxRadius);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(ccircles, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(ccircles, center, radius, Scalar(0, 255, 0), -1, 8, 0);
	}
	imshow("Hough_Circles", ccircles);
}