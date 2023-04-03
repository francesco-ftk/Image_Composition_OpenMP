#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void dataAugmentation(cv::Mat image, int transformations)
{
	for (int count = 0; count < transformations; ++count) {
		cv::Mat copy = image.clone();

		std::cout << "copy.rows: " << copy.rows << std::endl;
		std::cout << "copy.cols: " << copy.cols << std::endl;

		for (int i = 0; i < copy.rows; ++i) {
			for (int j = 0; j < copy.cols; ++j) {
				cv::Vec4b& pixel = image.at<cv::Vec4b>(i, j);
				pixel[0] = (pixel[0] + 255) / 2;
				pixel[1] = (pixel[1] + 255) / 2;
				pixel[2] = (pixel[2] + 255) / 2;
				pixel[3] = 128;

				/*
				std::cout << "pixel[0]: " << pixel[0] << std::endl;
				std::cout << "pixel[1]: " << pixel[1] << std::endl;
				std::cout << "pixel[2]: " << pixel[2] << std::endl;
				std::cout << "pixel[3]: " << pixel[3] << std::endl;*/
			}
		}

		cv::imwrite("../output/out.png", image);
		cv::imshow("Transparency", image);
		cv::waitKey(0);
	}
}

int main()
{
	double alpha = 0.5; double beta;

	cv::Mat src1, src2, dst;

	std::string src1_path = "../input/quokka.png";
	std::string src2_path = "../input/quokka2.png";

	std::cout << " Simple Linear Blender " << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "* Enter alpha [0.0-1.0]: ";
	//std::cin >> input;

	// We use the alpha provided by the user if it is between 0 and 1
	/*if( input >= 0 && input <= 1 )
	{ alpha = input; }*/

	src1 = cv::imread(src1_path);
	src2 = cv::imread(src2_path);

	if( src1.empty() ) { std::cout << "Error loading src1" << std::endl; return EXIT_FAILURE; }
	if( src2.empty() ) { std::cout << "Error loading src2" << std::endl; return EXIT_FAILURE; }

	beta = ( 1.0 - alpha );
	addWeighted(src1, alpha, src2, beta, 0.0, dst);

	cv::imshow( "Linear Blend", dst);
	cv::imwrite("../output/out.png", dst);

	cv::waitKey(0);

	/*std::string image_path = "../input/quokka.png";
	cv::Mat image = cv::imread(image_path);

	cv::Mat image_bgra;
	cvtColor(image, image_bgra, cv::COLOR_BGR2BGRA);

	if (image.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return 1;
	}
	cv::imshow("Original", image);

	dataAugmentation(image_bgra, 1);*/

    return 0;
}