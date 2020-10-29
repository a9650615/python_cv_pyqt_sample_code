import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

labels = []

with open("labelmap.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

interpreter = tflite.Interpreter(model_path="custom_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, output_details)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('w x h', width, height)

image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imH, imW, _ = image.shape
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

input_mean = 127.5
input_std = 127.5
input_data = (np.float32(input_data) - input_mean) / input_std #float32

interpreter.set_tensor(input_details[0]['index'],input_data)
interpreter.invoke()


boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects


print(boxes)

print(classes)

for i in range(len(scores)):
	if ((scores[i] > 0.3) and (scores[i] <= 1.0)):

		# Get bounding box coordinates and draw box
		# Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
		ymin = int(max(1,(boxes[i][0] * imH)))
		xmin = int(max(1,(boxes[i][1] * imW)))
		ymax = int(min(imH,(boxes[i][2] * imH)))
		xmax = int(min(imW,(boxes[i][3] * imW)))
		
		cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

		# Draw label
		object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
		label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
		labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
		label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
		cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
		cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

cv2.imwrite("output.jpg", image)