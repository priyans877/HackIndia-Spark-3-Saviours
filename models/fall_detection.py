import cv2
import mediapipe as mp
import numpy as np

class FallDetectionModel:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def calculate_angle(self, a, b, c):
        angle = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
        return np.abs(angle)

    def check_fall(self, results):
        if results.pose_landmarks:
            #extract relevant landmarks
            landmarks = results.pose_landmarks.landmark

            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            #calculate angles for detection
            shoulder_hip_angle = self.calculate_angle(left_shoulder, left_hip, right_hip)
            hip_knee_angle = self.calculate_angle(left_hip, left_knee, right_knee)
            knee_ankle_angle = self.calculate_angle(left_knee, left_ankle, right_ankle)

            #heuristic for detecting falls
            fall_from_stairs = shoulder_hip_angle > 1.5 and hip_knee_angle > 1 and knee_ankle_angle > 1.0
            unconscious_on_floor = self.check_unconscious(landmarks)

            if fall_from_stairs or unconscious_on_floor:
                return "Fall detected: " + ("Stairs" if fall_from_stairs else "Unconscious on floor")
        
        return "No fall detected"

    def check_unconscious(self, landmarks):
        #check unconscious
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        #check if shoulders and hips are aligned horizontally
        shoulder_angle = self.calculate_angle(left_shoulder, left_hip, right_hip)
        hip_angle = self.calculate_angle(left_hip, right_hip, left_hip)

        if shoulder_angle < 0.6 and hip_angle < 0.7:
            return True
        
        return False

    def draw_pose(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
