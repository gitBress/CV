#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>


class Classifier {
public:

	Classifier();

	std::string trainAndClassify(std::vector<cv::Mat> Images, std::vector<float> Map, int Classes, int Samples, std::vector<cv::Mat> chars);
	
};
