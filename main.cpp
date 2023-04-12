#include <iostream>
#include <filesystem>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace fs = std::filesystem;


int dataAugmentation(cv::Mat& foreground, std::vector<cv::Mat>& backgrounds, int transformations)
{
	std::srand(std::time(nullptr));
	std::time_t t = std::time(nullptr);
	std::tm* now = std::localtime(&t);

	std::stringstream folderName;
	folderName <<
			   (now->tm_year + 1900) << '-' <<
			   (now->tm_mon + 1) << '-' <<
			   now->tm_mday << '-' <<
			   now->tm_hour << '-' <<
			   now->tm_min << '-' <<
			   now->tm_sec;

	std::stringstream output_folder;
	output_folder << "../output/" << folderName.str() ;
	fs::create_directories(output_folder.str());

	for (int count = 0; count < transformations; ++count) {
		int index = (int) (std::rand() % backgrounds.size());
		cv::Mat background = backgrounds.at(index).clone();
		foreground = foreground.clone();

		if (background.rows < foreground.rows ||
			background.cols < foreground.cols)
		{
			std::cout << "ERROR: the foreground exceeds the background dimensions" << std::endl;
			return 1;
		}

		int row;
		if (background.rows - foreground.rows == 0) { row = 0; }
		else { row = (int) (std::rand() % (background.rows - foreground.rows)); }

		int col;
		if (background.cols - foreground.cols == 0) { col = 0; }
		else { col = (int) (std::rand() % (background.cols - foreground.cols)); }

		int alpha = 64 + rand() % 192;
		float alphaCorrectionFactor = (float) alpha / 255;
		float betaCorrectionFactor = 1 - alphaCorrectionFactor;


		for (int i = 0; i < foreground.rows; ++i)
		{
			for (int j = 0; j < foreground.cols; ++j)
			{
				cv::Vec4b& f_pixel = foreground.at<cv::Vec4b>(i, j);
				cv::Vec4b& b_pixel = background.at<cv::Vec4b>(row + i, col + j);
				float f_alpha = (float) f_pixel[3] / 255;
                if(f_alpha > 0.9){
                    b_pixel[0] = (unsigned char) ((float) b_pixel[0] * betaCorrectionFactor + (float) f_pixel[0] * alphaCorrectionFactor * f_alpha);
				    b_pixel[1] = (unsigned char) ((float) b_pixel[1] * betaCorrectionFactor + (float) f_pixel[1] * alphaCorrectionFactor * f_alpha);
				    b_pixel[2] = (unsigned char) ((float) b_pixel[2] * betaCorrectionFactor + (float) f_pixel[2] * alphaCorrectionFactor * f_alpha);
                }
			}
		}

		std::stringstream output_path;
		output_path << output_folder.str() << "/out_" << std::to_string(count) << ".png";
		cv::imwrite(output_path.str(), background);
	}
	return 0;
}

int main()
{
	std::string background_path = "../input/background";
	std::string foreground_path = "../input/foreground";

	// Read foreground
	const auto& f_entry = begin(fs::directory_iterator(foreground_path));
	cv::Mat foreground = cv::imread(f_entry->path().string(), cv::IMREAD_UNCHANGED);
	if (foreground.empty())
	{
		std::cout << "ERROR: foreground not found at: " << f_entry->path().string() << std::endl;
		return 1;
	}
	cvtColor(foreground, foreground, cv::COLOR_BGR2BGRA);

	// Read background
	std::vector<cv::Mat> backgrounds;
	for (const auto & b_entry : fs::directory_iterator(background_path)) {
		cv::Mat background = cv::imread(b_entry.path().string(), cv::IMREAD_UNCHANGED);
		if (!background.empty()) {
			cvtColor(background, background, cv::COLOR_BGR2BGRA);
			backgrounds.push_back(background);
		}
	}
	if (backgrounds.size() == 0)
	{
		std::cout << "ERROR: no backgrounds found at: " << background_path << std::endl;
		return 1;
	}

	cv::imshow("Foreground", foreground);
	cv::waitKey(0);

	dataAugmentation(foreground, backgrounds, 5);

    return 0;
}