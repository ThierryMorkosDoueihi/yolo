import numpy as np
import cv2
import os, sys, time

def no_overlap(cord1, cord2):
	return (cord1[2] < cord2[0] or cord2[2] < cord1[0] or cord1[3] < cord2[1] or cord2[3] < cord1[1])


args = {'confidence': 0.5, 'yolo': 'yolo-coco', 'threshold': 0.3, 'output': 'test.avi'}

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions

cap = cv2.VideoCapture(0)

(W, H) = (None, None)
ret,frame = cap.read()
counter = 0

# loop over frames from the video file stream
while(cap.isOpened()):
	# read the next frame from the file
	ret ,frame = cap.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not ret:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	
	

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > args["confidence"] and (classID == 0 or classID == 69):
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				if classID == 0:
					classIDs.append(classID)
				else:
					classIDs.append(1)


	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	bounds = [()]
	
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates to check overlap
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# store locations
			if classIDs[i] == 1:
				bounds[0] = (x,y,x+w,y+h)
			else:
				bounds.append((x,y,x+w,y+h))

	if len(bounds) > 1 and bounds[0] != ():
		flag = False
		for i in range(1,len(bounds)):
			flag = (flag or overlap(bounds[0],bounds[i]))
			if flag:
				break
		if not flag:
			if counter == 1:
				end = time.time()
				if (end - start)/60 > 5:
					print("ALERT!!!!!\nHAZARD!!!!!")
					counter = 0
			else:
				print("started timing")
				start = time.time()
				counter = 1
		else:
			print("overlapping")
			counter = 0
			

	cv2.imshow('Image',frame)
	
	k = cv2.waitKey(60) & 0xff
	if k == 27:
		break

# release the file pointers
cap.release()
