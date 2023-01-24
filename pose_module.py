import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode = False, model_complexity = 1, smooth_landmarks = True, enable_segmentation = False, smooth_segmentation = True,
                  detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode #whenever we create a new object it will have it's own variables
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)



    def findPose(self, img, draw = True): # will ask user if we want to draw or not
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img
    # for id, lm in enumerate(results.pose_landmarks.landmark):
    #         h, w, c = img.shape
    #         print(id, lm)
    #         cx, cy = int (lm.x * w), int(lm.y * h)
    #         cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
    #         #should overlay in previous points if we are detecting properly


def main():
    cap = cv2.VideoCapture('1.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (70,50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":  # if code running by itself it will call main function
    main()
