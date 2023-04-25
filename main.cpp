#include <iostream>
#include <filesystem>
#include <sstream>
#include <chrono>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

#define TRANSFORMATIONS 50


using namespace std::chrono;
namespace fs = std::filesystem;


int imageComposition(cv::Mat& foreground, std::vector<cv::Mat>& backgrounds, int transformations)
{
	// random initialization
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> uniform;

	// compute output folder name
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

	// create output folder
	std::stringstream outputFolderStream;
	outputFolderStream << "../output/" << folderName.str();
	std::string outputFolderPath = outputFolderStream.str();
	fs::create_directories(outputFolderPath);

	for (int count = 0; count < transformations; ++count)
	{
		// copy a random background from the shared pool
		int index = (int) (uniform(gen) % backgrounds.size());
		cv::Mat background = backgrounds.at(index).clone();

		// check the background dimensions
		if (background.rows < foreground.rows || background.cols < foreground.cols)
		{
			printf("ERROR: the foreground exceeds the background dimensions");
			break;
		}

		// pick a random position for the foreground
		int row;
		if (background.rows - foreground.rows == 0) { row = 0; }
		else { row = (int) (uniform(gen) % (background.rows - foreground.rows)); }
		int col;
		if (background.cols - foreground.cols == 0) { col = 0; }
		else { col = (int) (uniform(gen) % (background.cols - foreground.cols)); }

		// pick a random alpha value
		int alpha = 128 + uniform(gen) % 128;
		float alphaCorrectionFactor = (float) alpha / 255;
		float betaCorrectionFactor = 1 - alphaCorrectionFactor;

		//printf("Applying foreground...\n");
		for (int i = 0; i < foreground.rows; ++i)
		{
			for (int j = 0; j < foreground.cols; ++j)
			{
				cv::Vec4b &f_pixel = foreground.at<cv::Vec4b>(i, j);
				cv::Vec4b &b_pixel = background.at<cv::Vec4b>(row + i, col + j);
				float f_alpha = (float) f_pixel[3] / 255;
				if (f_alpha > 0.9)
				{
					b_pixel[0] = (unsigned char) ((float) b_pixel[0] * betaCorrectionFactor +
												  (float) f_pixel[0] * alphaCorrectionFactor * f_alpha);
					b_pixel[1] = (unsigned char) ((float) b_pixel[1] * betaCorrectionFactor +
												  (float) f_pixel[1] * alphaCorrectionFactor * f_alpha);
					b_pixel[2] = (unsigned char) ((float) b_pixel[2] * betaCorrectionFactor +
												  (float) f_pixel[2] * alphaCorrectionFactor * f_alpha);
				}
			}
		}

		std::string outputPath = outputFolderPath + "/out_" + std::to_string(count) + ".png";
		//printf("Saving at %s...\n", outputPath.c_str());
		cv::imwrite(outputPath, background);
		//printf("Iteration %d complete.\n", count);
	}
	return 0;
}

int imageCompositionOmp(cv::Mat& foreground, std::vector<cv::Mat>& backgrounds, int transformations)
{
	// random initialization
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> uniform;

	// compute output folder name
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

	// create output folder
	std::stringstream outputFolderStream;
	outputFolderStream << "../output/" << folderName.str();
	std::string outputFolderPath = outputFolderStream.str();
	fs::create_directories(outputFolderPath);

	#pragma omp parallel default(none) shared(foreground, backgrounds) firstprivate(transformations, outputFolderPath)
	{
		int threadId = omp_get_thread_num();
		//printf("Thread %d> Hello!\n", threadId);

		#pragma omp for
		for (int count = 0; count < transformations; ++count)
		{
			//printf("Thread %d> Starting iteration %d...\n", threadId, count);

			// copy a random background from the shared pool
			int index = (int) (uniform(gen) % backgrounds.size());
			cv::Mat background = backgrounds.at(index).clone();

			// check the background dimensions
			if (background.rows < foreground.rows || background.cols < foreground.cols)
			{
				printf("ERROR: the foreground exceeds the background dimensions");
				break;
			}

			// pick a random position for the foreground
			int row;
			if (background.rows - foreground.rows == 0) { row = 0; }
			else { row = (int) (uniform(gen) % (background.rows - foreground.rows)); }
			int col;
			if (background.cols - foreground.cols == 0) { col = 0; }
			else { col = (int) (uniform(gen) % (background.cols - foreground.cols)); }

			// pick a random alpha value
			int alpha = 128 + uniform(gen) % 128;
			float alphaCorrectionFactor = (float) alpha / 255;
			float betaCorrectionFactor = 1 - alphaCorrectionFactor;

			//printf("Thread %d> Applying foreground...\n", threadId);
			for (int i = 0; i < foreground.rows; ++i)
			{
				for (int j = 0; j < foreground.cols; ++j)
				{
					cv::Vec4b &f_pixel = foreground.at<cv::Vec4b>(i, j);
					cv::Vec4b &b_pixel = background.at<cv::Vec4b>(row + i, col + j);
					float f_alpha = (float) f_pixel[3] / 255;
					if (f_alpha > 0.9)
					{
						b_pixel[0] = (unsigned char) ((float) b_pixel[0] * betaCorrectionFactor +
													  (float) f_pixel[0] * alphaCorrectionFactor * f_alpha);
						b_pixel[1] = (unsigned char) ((float) b_pixel[1] * betaCorrectionFactor +
													  (float) f_pixel[1] * alphaCorrectionFactor * f_alpha);
						b_pixel[2] = (unsigned char) ((float) b_pixel[2] * betaCorrectionFactor +
													  (float) f_pixel[2] * alphaCorrectionFactor * f_alpha);
					}
				}
			}

			std::string outputPath = outputFolderPath + "/out_" + std::to_string(count) + ".png";
			//printf("Thread %d> Saving at %s...\n", threadId, outputPath.c_str());
			cv::imwrite(outputPath, background);
			//printf("Thread %d> Iteration %d complete.\n", threadId, count);
		}
	}
	return 0;
}

int imageCompositionOmpFineGrain(cv::Mat& foreground, std::vector<cv::Mat>& backgrounds, int transformations)
{
	// random initialization
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> uniform;

	// compute output folder name
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

	// create output folder
	std::stringstream outputFolderStream;
	outputFolderStream << "../output/" << folderName.str();
	std::string outputFolderPath = outputFolderStream.str();
	fs::create_directories(outputFolderPath);

	int row;
	int col;
	int alpha;
	cv::Mat background;

	#pragma omp parallel default(none) shared(foreground, backgrounds, background, uniform, gen, row, col, alpha) firstprivate(transformations, outputFolderPath)
	{
		int threadId = omp_get_thread_num();
		//printf("Thread %d> Hello!\n", threadId);

		for (int count = 0; count < transformations; ++count)
		{
			#pragma omp single
			{
				//printf("Thread %d> Starting iteration %d...\n", threadId, count);

				// copy a random background from the shared pool
				int index = (int) (uniform(gen) % backgrounds.size());
				background = backgrounds.at(index).clone();

				// check the background dimensions
				if (background.rows < foreground.rows || background.cols < foreground.cols) {
					printf("ERROR: the foreground exceeds the background dimensions");
				}

				// pick a random position for the foreground
				if (background.rows - foreground.rows == 0) { row = 0; }
				else { row = (int) (uniform(gen) % (background.rows - foreground.rows)); }
				if (background.cols - foreground.cols == 0) { col = 0; }
				else { col = (int) (uniform(gen) % (background.cols - foreground.cols)); }

				// pick a random alpha value
				alpha = 128 + uniform(gen) % 128;
			}
			float alphaCorrectionFactor = (float) alpha / 255;
			float betaCorrectionFactor = 1 - alphaCorrectionFactor;

			//printf("Thread %d> Arrived at for loop\n", threadId);
			#pragma omp barrier
			for (int i = 0; i < foreground.rows; ++i)
			{
				/*if (i == 0)
				{
					printf("Thread %d> Working on image...\n", threadId);
				}*/
				#pragma omp for
				for (int j = 0; j < foreground.cols; ++j)
				{
					//printf("Thread %d> Working on pixel\t[%d,%d]...\n", threadId, i, j);
					cv::Vec4b &f_pixel = foreground.at<cv::Vec4b>(i, j);
					cv::Vec4b &b_pixel = background.at<cv::Vec4b>(row + i, col + j);
					float f_alpha = (float) f_pixel[3] / 255;
					if (f_alpha > 0.9)
					{
						b_pixel[0] = (unsigned char) ((float) b_pixel[0] * betaCorrectionFactor +
													  (float) f_pixel[0] * alphaCorrectionFactor * f_alpha);
						b_pixel[1] = (unsigned char) ((float) b_pixel[1] * betaCorrectionFactor +
													  (float) f_pixel[1] * alphaCorrectionFactor * f_alpha);
						b_pixel[2] = (unsigned char) ((float) b_pixel[2] * betaCorrectionFactor +
													  (float) f_pixel[2] * alphaCorrectionFactor * f_alpha);
					}
				}
			}

			#pragma omp single
			{
				std::string outputPath = outputFolderPath + "/out_" + std::to_string(count) + ".png";
				//printf("Thread %d> Saving at %s...\n", threadId, outputPath.c_str());
				cv::imwrite(outputPath, background);
				//printf("Thread %d> Iteration %d complete.\n", threadId, count);
			}
		}
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

	auto start_time = high_resolution_clock::now();
	imageCompositionOmp(foreground, backgrounds, TRANSFORMATIONS);
	auto end_time = high_resolution_clock::now();

	float totalTime = (float) duration_cast<microseconds>(end_time - start_time).count() / 1000.f;

	printf("Using # %d threads.\n", omp_get_max_threads());
	printf("Total time:   %fms\n", totalTime);
	printf("Number of output images: %d\n", TRANSFORMATIONS);

    return 0;
}