#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

//using namespace cv;

	// constructor
	Filter::Filter(cv::Mat input_img, int size) {

		input_image = input_img;
		if (size % 2 == 0)
			size++;
		filter_size = size;
	}

	// for base class do nothing (in derived classes it performs the corresponding filter)
	void Filter::doFilter() {

		// it just returns a copy of the input image
		result_image = input_image.clone();

	}

	// get output of the filter
	cv::Mat Filter::getResult() {

		return result_image;
	}

	//set window size (it needs to be odd)
	void Filter::setSize(int size) {

		if (size % 2 == 0)
			size++;
		filter_size = size;
	}

	//get window size 
	int Filter::getSize() {

		return filter_size;
	}



	// Write your code to implement the Gaussian, median and bilateral filters

	//Gaussian

	GaussianFilter::GaussianFilter(cv::Mat input_img, int size, int sigma):Filter(input_img, size) {
		sigmaX = static_cast<double>(sigma);
	}

	void GaussianFilter::doFilter() {
		GaussianBlur(input_image, result_image, cv::Size(filter_size, filter_size), sigmaX, 0, cv::BORDER_DEFAULT);
	}

	void GaussianFilter::changeSigma(int sigma) {
		sigmaX = static_cast<double>(sigma);
	}

	//Median

	MedianFilter::MedianFilter(cv::Mat input_img, int size) :Filter(input_img, size) {

	}

	void MedianFilter::doFilter() {
		medianBlur(input_image, result_image, filter_size);
	}

	//Bilateral

	BilateralFilter::BilateralFilter(cv::Mat input_img, int size, int sigmaRange, int sigmaSpace) :Filter(input_img, size) {
		sigma_range = static_cast<double>(sigmaRange);
		sigma_space = static_cast<double>(sigmaSpace);
	}

	void BilateralFilter::doFilter() {
		bilateralFilter(input_image, result_image, filter_size, sigma_range, sigma_space);
	}

	void BilateralFilter::changeSigmaRange(int sigmaRange) {
		sigma_range = static_cast<double>(sigmaRange);
	}

	void BilateralFilter::changeSigmaSpace(int sigmaSpace) {
		sigma_space = static_cast<double>(sigmaSpace);
	}

	