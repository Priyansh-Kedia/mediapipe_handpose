import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def euclidean(point1, point2):
    dist = np.linalg.norm(point1 - point2)
    return dist

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        point1 = np.array((hand_landmarks.landmark[12].x,hand_landmarks.landmark[12].y,hand_landmarks.landmark[12].z))
        point2 = np.array((hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y,hand_landmarks.landmark[8].z))
        fingers = abs(euclidean(point1, point2))
        point1 = np.array((hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y,hand_landmarks.landmark[4].z))
        point2 = np.array((hand_landmarks.landmark[16].x,hand_landmarks.landmark[16].y,hand_landmarks.landmark[16].z))
        close = abs(euclidean(point1, point2))
        # print(fingers, close)
        if fingers > 0.08 and close < 0.035:
            print("Number is 2", close)
            break
        # print(len(hand_landmarks.landmark), euclidean(point1, point2))
        # for idx, hl in enumerate(hand_landmarks.landmark):
        #     print(hl)
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()