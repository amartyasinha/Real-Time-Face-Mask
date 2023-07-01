# imports
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist
import argparse
import imutils
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import requests


FLAGS = []

# face mask detection function
def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (MstartX, MstartY, MendX, MendY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (MstartX, MstartY) = (max(0, MstartX), max(0, MstartY))
            (MendX, MendY) = (min(w - 1, MendX), min(h - 1, MendY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[MstartY:MendY, MstartX:MendX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((MstartX, MstartY, MendX, MendY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights',
                        default='./weights/yolov3.weights')
    parser.add_argument('-cfg', '--config',
                        default='./yolo-coco/yolov3.cfg')
    parser.add_argument('-v', '--video-path',
                        default='pedestrians.mp4')
    parser.add_argument('-vo', '--video-output-path',
                        default='output.avi')
    parser.add_argument('-d', '--display',
                        default=1)
    parser.add_argument('-l', '--labels',
                        default='./yolo-coco/coco.names')

    FLAGS, unparsed = parser.parse_known_args()

    # Get the labels
    LABELS = open(FLAGS.labels).read().strip().split('\n')

    # Load the weights and configuration to form the pretrained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

if FLAGS.video_path:
    # initialize the video stream and pointer to output video file
    # load our serialized face detector model from disk
    prototxtPath = r"face_detector/deploy.prototxt"
    weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    #url = "http://192.168.1.9:8080/shot.jpg"


    # loop over the frames from the video stream
    while True:
        #img_resp = requests.get(url)
        #img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
        #frame = cv2.imdecode(img_arr, -1)

        # read the next frame from the input video
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then that's the end fo the stream 
        # if not grabbed:
        #     break

        # resize the frame and then detect people (only people) in it
        frame = imutils.resize(frame, width=1000, height=1000)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (MstartX, MstartY, MendX, MendY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (MstartX, MstartY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (MstartX, MstartY), (MendX, MendY), color, 2)

        # initialize the set of indexes that violate the minimum social distance
        violate = set()

        # ensure there are at least two people detections (required in order to compute the
        # the pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the Euclidean distances
            # between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two centroid pairs is less
                    # than the configured number of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update the violation set with the indexes of the centroid pairs
                        violate.add(i)
                        violate.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # draw the total number of social distancing violations on the output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # check to see if the output frame should be displayed to the screen
        if FLAGS.display > 0:
            # show the output frame
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break
