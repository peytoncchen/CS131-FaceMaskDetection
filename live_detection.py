import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

PATH = './face_mask.pth'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, (3,3))
        self.conv2 = nn.Conv2d(32, 32, (3,3))
        self.conv3 = nn.Conv2d(32, 32, (3,3))
        
        self.maxpool = nn.MaxPool2d((3,3))
        
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


model = Net()
model.load_state_dict(torch.load(PATH))
prototxt = 'deploy.prototxt'
weights = 'SSD.caffemodel'
faceNet = cv2.dnn.readNet(prototxt, weights)

vs = VideoStream(src = 0).start()

def detect_and_predict_mask(frame, faceNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces, locs, preds = [], [], []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# filter out weak detections 
		if confidence > 0.4:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			locs.append((startX, startY, endX, endY))
			faces.append(face)

	# only make prediction if there are actually faces
	if len(faces) > 0:
		for face in faces:
			cv2.imwrite('temp.jpg',face)
			image = Image.open('temp.jpg')
			transform = transforms.Compose([transforms.Resize((150, 150)), 
							transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
			transformed_img = transform(image)
			result=model(transformed_img[None, ...])
			preds.append(int(torch.round(result)))

	return (locs, preds)


while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	(locs, preds) = detect_and_predict_mask(frame, faceNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		label = "Mask" if pred == 0 else "No Mask"
		
		color = (0, 255, 0) if pred == 0 else (0, 0, 255)

		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Peyton's M1 MBP", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()