# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

def getImgName(path):
	from PIL import Image
	import glob
	image_list = []
	for filename in glob.glob(path + '/*.jpg'):
		image_list.append(filename)
	return image_list

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

path = './images'
image_list = getImgName(path)
print(image_list)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


def saveAlignFaces(imgPath, detector, predictor):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor and the face aligner
	#imgPath = './images/' + imgName

	fa = FaceAligner(predictor, desiredFaceWidth=256)
	
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(imgPath)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale
	# image
	cv2.imshow("Input", image)
	rects = detector(gray, 2)

	# loop over the face detections
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rect)

		import uuid
		#f = str(uuid.uuid4())
		imgName  = imgPath[9:]
		# display the output images
		cv2.imshow("Original", faceOrig)
		print(faceAligned)
		cv2.imshow("Aligned", faceAligned)
		cv2.imwrite( "./result/" + imgName, faceAligned );
		#cv2.waitKey(0)
		
for i in image_list:
	saveAlignFaces(i, detector, predictor)