import numpy as np
import dlib
import cv2
import utils 

DEFAULT_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

class FacialAnalysis:
	def __init__(self,predictor_path=DEFAULT_PREDICTOR_PATH):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(predictor_path)

	def DrawLandmarks(self,image):
		'''
		@parms: grayscale image
		Predict the lanmarks on image and return the image.
		'''
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale image
		rects = self.detector(image, 1)

		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = self.predictor(gray, rect)
			shape = utils.shape_to_np(shape)
		 
			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = utils.rect_to_bb(rect)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		 
			# show the face number
			cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		 
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		return image

fa = FacialAnalysis()

image = cv2.imread("sample.png")

fa.DrawLandmarks(image)
cv2.imshow("Output", image)
cv2.waitKey(0)
 


