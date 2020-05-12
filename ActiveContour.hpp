#ifndef _ACTIVE_CONTOUR_H
#define _ACTIVE_CONTOUR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// List of directions and step size
const int STEP_SIZE = 1; 
const int DIR_CNT = 8; //4;
const Point dirs[] = {
	Point(0,-STEP_SIZE),
	Point(0, STEP_SIZE),
	Point(STEP_SIZE, 0),
	Point(-STEP_SIZE,0),

	Point(-STEP_SIZE,-STEP_SIZE),
	Point(STEP_SIZE,-STEP_SIZE),
	Point(-STEP_SIZE,STEP_SIZE),
	Point(STEP_SIZE,STEP_SIZE)
};

// Functions
void computeExternalEnergyImage(Mat input, Mat& energyImage);
double computeEnergyForCurve(vector<Point>& allPoints, Mat & energyImage, double iaw, double ibw, double ew);
bool updateCurve(vector<Point>& allPoints, Mat& energyImage, double iaw, double ibw, double ew);

#endif
