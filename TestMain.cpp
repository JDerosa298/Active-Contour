#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include "ActiveContour.hpp"

using namespace cv;
using namespace std;

/////////////////////////////////////////////////
// Globals
/////////////////////////////////////////////////

vector<Point> snaxels;
int pointIndexToUpdate = -1;
bool leftMouseDown = false;
const double MAX_NEAR_POINT = 10.0;
const int POINT_DRAW_RADIUS = 4;
const int LINE_DRAW_WIDTH = 2;

const double INTERNAL_WEIGHT_A = 1.0;
const double INTERNAL_WEIGHT_B = 2.0;
const double EXTERNAL_WEIGHT = 1.0;

/////////////////////////////////////////////////
// Functions
/////////////////////////////////////////////////

int nearestPoint(int x, int y, double maxDist) {
	int index = -1;
	double minDist = -1;
	Point click = Point(x, y);
	for (int i = 0; i < snaxels.size(); i++) {
		double dist = cv::norm(snaxels.at(i) - click);
		if (dist <= maxDist &&
			(minDist < 0 || dist < minDist)) {
			minDist = dist;
			index = i;
		}
	}

	return index;
}

void mouseFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN) {
		// Are we near an existing point?
		int nearIndex = nearestPoint(x, y, MAX_NEAR_POINT);
		if (nearIndex >= 0) {
			// Set point to update to old point
			pointIndexToUpdate = nearIndex;
		}
		else {
			// Add new point
			snaxels.push_back(Point(x, y));
			// Set point to update
			pointIndexToUpdate = ((int)snaxels.size()) - 1;
		}

		// Left mouse button down
		leftMouseDown = true;
	}
	else if (event == EVENT_MOUSEMOVE) {
		// Is the left mouse button down?
		if (leftMouseDown) {
			// Move point
			snaxels.at(pointIndexToUpdate).x = x;
			snaxels.at(pointIndexToUpdate).y = y;
		}
	}
	else if (event == EVENT_LBUTTONUP) {
		leftMouseDown = false;
	}
}

void drawSnaxels(Mat image) {
	// Draw lines
	for (int i = 1; i < snaxels.size(); i++) {
		Point p1 = snaxels.at(i-1);
		Point p2 = snaxels.at(i);
		line(image, p1, p2, Scalar(0, 255, 0), LINE_DRAW_WIDTH);
	}

	// Draw points
	for (int i = 0; i < snaxels.size(); i++) {
		Point p = snaxels.at(i);
		circle(image, p, POINT_DRAW_RADIUS, Scalar(0, 0, 255), -1);
	}
}

int main(int argc, char** argv) {

	// If no command line arguments entered, use webcam
	if (argc <= 1) {
		// Webcam

		cout << "Opening webcam..." << endl;

		// Grab the default camera
		VideoCapture camera(0);

		// Did we get it?
		if (!camera.isOpened()) {
			cout << "ERROR: Cannot open camera!" << endl;
			return -1;
		}

		// Create window ahead of time
		string windowName = "Webcam";
		namedWindow(windowName);

		// Set mouse callback
		setMouseCallback(windowName, mouseFunc, NULL);
		
		int key = -1;
		Mat frame;
		Mat grayImage;
		Mat canvasImage;
		Mat energyImage;
		Mat displayEnergy;
		bool updatingCurve = false;

		// While the key isn't the escape key...
		while (key != 27) {
			
			// Get next frame from camera
			camera >> frame;

			// Convert to grayscale
			cvtColor(frame, grayImage, COLOR_BGR2GRAY);

			// Compute energy image
			computeExternalEnergyImage(grayImage, energyImage);

			// Convert BACK to color (just for display)
			cvtColor(grayImage, canvasImage, COLOR_GRAY2BGR);

			// Draw current points
			drawSnaxels(canvasImage);

			// Show the image
			imshow(windowName, canvasImage);
			cv::log(energyImage, displayEnergy);
			cv::normalize(displayEnergy, displayEnergy, 1.0, 0.0, cv::NORM_MINMAX);
			imshow("External Energy", displayEnergy);

			// Wait 30 milliseconds, and grab any key presses
			key = waitKey(30);

			// Check for key commands
			if (key == 'r') {
				cout << "RESET POINTS." << endl;
				snaxels.clear();
			}
			else if (key == ' ') {
				updatingCurve = !updatingCurve;
			}
			
			// Are we updating the curve?
			if (updatingCurve) {
				// Update points
				if (!updateCurve(
					snaxels,
					energyImage,
					INTERNAL_WEIGHT_A,
					INTERNAL_WEIGHT_B,
					EXTERNAL_WEIGHT)) {
					cout << "No update." << endl;
					updatingCurve = false;
				}
				cout << "Current energy: ";
				double energy = computeEnergyForCurve(
					snaxels,
					energyImage,
					INTERNAL_WEIGHT_A,
					INTERNAL_WEIGHT_B,
					EXTERNAL_WEIGHT);
				cout << energy << endl;
			}
		}

		// Camera's destructor will close the camera

		cout << "Closing application..." << endl;
	}
	else {
		// Try to load image from argument

		// Get filename
		string filename = string(argv[1]);

		// Load image
		cout << "Loading image: " << filename << endl;
		Mat image = imread(filename); // For grayscale: imread(filename, IMREAD_GRAYSCALE);

		// Check if data is invalid
		if (!image.data) {
			cout << "ERROR: Could not open or find the image!" << endl;
			return -1;
		}

		// Create window
		namedWindow(filename);

		// Set mouse callback
		setMouseCallback(filename, mouseFunc, NULL);
		
		int key = -1;
		Mat grayImage;
		Mat canvasImage;
		
		// Convert to grayscale
		cvtColor(image, grayImage, COLOR_BGR2GRAY);

		// Convert BACK to color
		cvtColor(grayImage, image, COLOR_GRAY2BGR);
			   
		// Compute energy image
		Mat energyImage;
		Mat displayEnergy;
		computeExternalEnergyImage(grayImage, energyImage);
		
		// While key isn't the escape key
		bool updatingCurve = false;
		while (key != 27) {
			// Recopy original image for display
			image.copyTo(canvasImage);
			
			// Draw current points
			drawSnaxels(canvasImage);

			// Show our image (with the filename as the window title)
			imshow(filename, canvasImage);
			cv::log(energyImage, displayEnergy);			
			cv::normalize(displayEnergy, displayEnergy, 1.0, 0.0, cv::NORM_MINMAX);
			imshow("External Energy", displayEnergy); 

			// Wait 30 milliseconds
			key = waitKey(30);

			// Check for key commands
			if (key == 'r') {
				cout << "RESET POINTS." << endl;
				snaxels.clear();
			}
			else if (key == ' ') {
				updatingCurve = !updatingCurve;
			}

			// Are we updating the curve?
			if (updatingCurve) {
				// Update points
				if (!updateCurve(
					snaxels,
					energyImage,
					INTERNAL_WEIGHT_A,
					INTERNAL_WEIGHT_B,
					EXTERNAL_WEIGHT)) {
					cout << "No update." << endl;
					updatingCurve = false;
				}
				cout << "Current energy: ";
				double energy = computeEnergyForCurve(
					snaxels,
					energyImage,
					INTERNAL_WEIGHT_A,
					INTERNAL_WEIGHT_B,
					EXTERNAL_WEIGHT);
				cout << energy << endl;
			}
		}

		// Cleanup this window
		destroyWindow(filename);
		// If we wanted to get rid of ALL windows: destroyAllWindows();			
	}

	return 0;
}
