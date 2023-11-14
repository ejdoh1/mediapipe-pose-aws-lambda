"""
AWS Lambda function to extract pose landmarks from an image and 
return a presigned url to the annotated image
"""

import json
import os
import uuid

import boto3
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.IMAGE,
)
landmarker = PoseLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws the landmarks and the connections on the image."""

    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def handler(_event, _context):
    """
    Extracts pose landmarks from an image and returns a presigned url to the annotated image.
    For demo the image is fixed to image.jpg.
    """

    mp_image = mp.Image.create_from_file("image.jpg")

    # start timer
    timer = cv2.getTickCount()
    pose_landmarker_result = landmarker.detect(mp_image)
    # stop timer
    time_in_ms = (cv2.getTickCount() - timer) / cv2.getTickFrequency() * 1000
    print("Time taken to run pose landmark detection: %.3f ms" % time_in_ms)

    print(pose_landmarker_result)

    # Draw pose landmarks on the image.
    # start timer
    timer = cv2.getTickCount()
    annotated_image = draw_landmarks_on_image(
        mp_image.numpy_view(), pose_landmarker_result
    )
    # stop timer
    time_in_ms = (cv2.getTickCount() - timer) / cv2.getTickFrequency() * 1000

    print("Time taken to draw pose landmarks: %.3f ms" % time_in_ms)

    # start timer
    timer = cv2.getTickCount()

    filename = str(uuid.uuid4()) + ".png"
    filepath = "/tmp/" + filename

    cv2.imwrite(filepath, annotated_image)

    # upload to s3 and return a presigned url
    s3 = boto3.client("s3")
    s3_bucket = os.environ["S3_BUCKET"]
    s3.upload_file(filepath, s3_bucket, filename)
    s3_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": s3_bucket, "Key": filename},
        ExpiresIn=3600,
    )
    # stop timer
    time_in_ms = (cv2.getTickCount() - timer) / cv2.getTickFrequency() * 1000

    print("Time taken to upload to s3: %.3f ms" % time_in_ms)

    return {
        "statusCode": 200,
        "body": json.dumps({"url": s3_url, "expires_in": 3600}),
    }


if __name__ == "__main__":
    handler(None, None)
