# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
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
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-pr", "--procent", type=float, default=0.65,
	help="procent confidence")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
predictor = dlib.shape_predictor(args["shape_predictor"])
#detector111 = dlib.get_frontal_face_detector()
detector = YOLO()

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
vs = vs = cv2.VideoCapture(args["input"])
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

writer = None
h, w = None, None


# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame
	#print(np.shape(frame))
	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break
	frame = imutils.resize(frame, width=800)
	
	if w is None or h is None:
		(h, w) = frame.shape[:2]
	
	if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(w, h), True)
	
	
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	

	detections = []
	rects = []
	
	# load the input image, resize it, and convert it to grayscale
	
	
	out_boxes, out_scores, out_classes = detect_image(detector, Image.fromarray(frame))
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
	
	fa = FaceAligner(predictor, desiredFaceWidth=256)	
	
	#rects = detector(gray, 2)

	for rect in rects:
		faceAligned = fa.align(frame, gray, rect)
		#print(faceAligned)
		
		#try to rise resolution
		#grayFaceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)	
		#blurred = cv2.GaussianBlur(grayFaceAligned, (5, 5), 0)
		#faceAligned = blurred
		#clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
		#faceAligned = clahe.apply(faceAligned)
	
		#faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_GRAY2RGB)
		
		cv2.imshow("Aligned", np.asarray(faceAligned))
		#cv2.waitKey(0)
		detections.append(faceAligned)

	
	# loop over the detections
	for i, face in enumerate(detections):
		(fH, fW) = face.shape[:2]
		(x, y, x1, y1) = out_boxes[i]
		w = abs(x - x1)
		h = abs(y - y1)
		
		startX =  int(min(x1, x))
		endX =  int(startX + w)
		startY = int(min(y1, y))
		endY = int(startY + h)
		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
			(96, 96), (0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]
		if (proba > args["procent"]):
			#print(proba * 100, args["procent"])
			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			if (name != 'unknown'):
				cv2.rectangle(frame, (startY, startX), (endY, endX),
				(120, 100, 50), 2)
				cv2.putText(frame, text, (y, startX),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 100, 50), 2)
			else:
				cv2.rectangle(frame, (startY, startX), (endY, endX),
				(0, 0, 255), 2)
				cv2.putText(frame, text, (y, startX),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		else:
			cv2.rectangle(frame, (startY, startX), (endY, endX),
			(0, 199, 255), 2)
			cv2.putText(frame, "I'm not sure", (y, startX),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 199, 255), 2)
	# update the FPS counter
	fps.update()
	fps.stop()
	#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.putText(frame, "[INFO] fps: {:.2f}".format(fps.fps()), (10, 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 170, 255), 2)
	fps = FPS().start()
	# show the output frame
	
	if writer is not None:
		writer.write(frame)
			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	
			
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()
	
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()