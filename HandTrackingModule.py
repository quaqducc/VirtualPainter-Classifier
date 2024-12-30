import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.85, trackCon = 0.5):
        #print("Khởi tạo handDetector...")
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        
        #print("Đã khởi tạo mp.solutions.hands...")
        
        # self.mode, self.maxHands, self.detectionCon, self.trackCon
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, min_detection_confidence = self.detectionCon, min_tracking_confidence=self.trackCon)
        #print("Đã khởi tạo self.hands...")

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIDs = [4,8,12,16,20]
        #print("Đã khởi tạo mpDraw...")
        
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
        return self.lmList
    
    def fingerUp(self):
        
        if not hasattr(self, 'lmList') or len(self.lmList) == 0:
            return [0, 0, 0, 0, 0] 
        fingers = []
        
        # Thumb
        if self.lmList[self.tipIDs[0]][1] < self.lmList[self.tipIDs[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers
        for id in range(1,5):
            if self.lmList[self.tipIDs[id]][2] < self.lmList[self.tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    if not cap.isOpened():
        print("Không thể mở webcam!")
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        print(f"results.multi_hand_landmarks: {detector.results.multi_hand_landmarks}")
        lmList = detector.findPosition(img)
        
        print(len(lmList))
        if len(lmList) != 0:
            print(lmList[4])
            #cv2.circle(img, (self.lmList[8][1], self.lmList[8][2]), 15, (255, 0, 255), cv2.FILLED)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
if __name__ == '__main__':
    main()