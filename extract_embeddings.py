# USAGE
# python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
#	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import dlib
from PIL import Image
from yolo import YOLO, detect_video
from yolo3.utils import letterbox_image
from keras import backend as K

def detect_image(self, image):
    if self.model_image_size != (None, None):
        assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    #print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = self.sess.run(
       [self.boxes, self.scores, self.classes],
       feed_dict={
            self.yolo_model.input: image_data,
            self.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    return out_boxes, out_scores, out_classes

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
predictor = dlib.shape_predictor(args["shape_predictor"])
#detector = dlib.get_frontal_face_detector()
detector = YOLO()

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 800 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=800)
	
	#try to rise resolution
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	#image = blurred
	#clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
	#image = clahe.apply(image)
	
	#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	
	(h, w) = image.shape[:2]
	

	# we're making the assumption that each image has only ONE
	# face, so find the bounding box with the largest probability

	
	#align_faces
	fa = FaceAligner(predictor, desiredFaceWidth=256)	
	
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#rects = detector(gray, 2)
	rects = []
	
	out_boxes, out_scores, out_classes = detect_image(detector, Image.fromarray(image))
	for i, c in reversed(list(enumerate(out_classes))):
		(x, y, x1, y1) = out_boxes[i]
		w = abs(x - x1)
		h = abs(y - y1)
		
		startX =  int(min(x1, x))
		endX =  startX + w
		startY = int(min(y1, y))
		endY = startY + h
		left, right, bottom, top  = startX, endX, endY, startY
		
		rect = dlib.rectangle(int(top), int(left), int(bottom) , int(right))
		rects.append(rect)
	
	
	for rect in rects:
		faceAligned = fa.align(image, gray, rect)
		print(faceAligned)
		cv2.imshow("Aligned", np.asarray(faceAligned))
		cv2.waitKey(0)
	face = faceAligned
			
	(fH, fW) = face.shape[:2]
			
	# ensure the face width and height are sufficiently large
	if fW < 20 or fH < 20:
		continue

	# construct a blob for the face ROI, then pass the blob
	# through our face embedding model to obtain the 128-d
	# quantification of the face
	faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
		(96, 96), (0, 0, 0), swapRB=True, crop=False)
	embedder.setInput(faceBlob)
	vec = embedder.forward()

	# add the name of the person + corresponding face
	# embedding to their respective lists
	knownNames.append(name)
	knownEmbeddings.append(vec.flatten())
	total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()