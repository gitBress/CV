#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>

class FindChar {
public:

	FindChar(cv::Mat plate);

	void preprocCHAR();
	void contPlate();
	bool FindChar::maybeChar(std::vector<cv::Point> cont);
	cv::Mat FindChar::extractChar(cv::Mat origImg, std::vector<cv::Point> oneChar);

	cv::Mat preprocPlate;
	std::vector<std::vector<cv::Point>> contours;


protected:

	cv::Mat origPlate;
	
};
