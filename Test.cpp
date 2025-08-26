#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>
#include<iostream>

using namespace std;
using namespace cv;

void main() {
	VideoCapture video(0);
	Mat img;

	CascadeClassifier faceCascade;
	faceCascade.load("haarcascade_frontalface_default.xml");


	while (true) {
		video.read(img);

		vector<Rect> faces;
		faceCascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(30, 30));

		for (int i = 0; i < faces.size(); i++) {
			rectangle(img, faces[i], Scalar(255, 0, 255), 2);
		}

		imshow("Frame...", img);

		waitKey(1);
	}
}