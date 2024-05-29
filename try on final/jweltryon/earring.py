import mediapipe as mp
import cv2
import numpy as np

#checked with six images

def get_landmark_position(face_landmarks, landmark_index):
    landmark = face_landmarks.landmark[landmark_index]
    return landmark.x,landmark.y,landmark.z


def overlay_earrings(image, face_landmarks,suitable_earring, selected_indices):
    selected_earring_index = selected_indices
    selected_earring_image = suitable_earring[selected_earring_index]
    left_ear_landmark_index = 234
    right_ear_landmark_index = 454

# 0.42066726326942444
    treshold_left = 0.4099497985839844
    treshold_right = 0.4723720288276672

    current_position = get_landmark_position(face_landmarks, 0)
    left_ear_x, left_ear_y = int(face_landmarks.landmark[left_ear_landmark_index].x * image.shape[1]), int(
        face_landmarks.landmark[left_ear_landmark_index].y * image.shape[0])
    right_ear_x, right_ear_y = int(face_landmarks.landmark[right_ear_landmark_index].x * image.shape[1]), int(
        face_landmarks.landmark[right_ear_landmark_index].y * image.shape[0])


    earring_height = 67
    earring_width = 24

    # Landmark 0 (x, y, z): 0.5224074721336365 0.6143783330917358 -0.031618259847164154
    # Landmark 0 (x, y, z): 0.5350976586341858 0.6374034881591797 -0.02616993896663189
    tresh_z = -0.025618259847164154


    selected_earring_image = cv2.resize(selected_earring_image, (earring_width, earring_height))

    if left_ear_x > 0 and left_ear_y > 0 and right_ear_x > 0 and right_ear_y > 0:
        left_earring_x_start = left_ear_x - int(earring_width / 2) - 3
        left_earring_y_start = left_ear_y + int(earring_height / 4) + 8

        left_earring_x_end = left_earring_x_start + earring_width
        left_earring_y_end = left_earring_y_start + earring_height

        right_earring_x_start = right_ear_x - int(earring_width / 2) + 3
        right_earring_y_start = right_ear_y + int(earring_height / 4) + 8

        right_earring_x_end = right_earring_x_start + earring_width
        right_earring_y_end = right_earring_y_start + earring_height

        resized_left_earring = cv2.resize(selected_earring_image, (earring_width, earring_height))
        alpha_s_left = resized_left_earring[:, :, 3] / 255.0
        alpha_l_left = 1.0 - alpha_s_left

        resized_right_earring = cv2.resize(selected_earring_image, (earring_width, earring_height))
        alpha_s_right = resized_right_earring[:, :, 3] / 255.0
        alpha_l_right = 1.0 - alpha_s_right

        if current_position[0]>treshold_left and current_position[0]<treshold_right :
            try:
                for c in range(3):
                    image[left_earring_y_start:left_earring_y_end, left_earring_x_start:left_earring_x_end, c] = (
                                    alpha_s_left * resized_left_earring[:, :, c] +
                                    alpha_l_left * image[left_earring_y_start:left_earring_y_end,
                                    left_earring_x_start:left_earring_x_end, c]
                            )

                    image[right_earring_y_start:right_earring_y_end, right_earring_x_start:right_earring_x_end, c] = (
                                    alpha_s_right * resized_right_earring[:, :, c] +
                                    alpha_l_right * image[right_earring_y_start:right_earring_y_end,
                                    right_earring_x_start:right_earring_x_end, c]
                            )
            except ValueError:
                pass

        elif current_position[0]>treshold_left:
            try:
                for c in range(3):
                    image[left_earring_y_start:left_earring_y_end, left_earring_x_start:left_earring_x_end, c] = (
                                    alpha_s_left * resized_left_earring[:, :, c] +
                                    alpha_l_left * image[left_earring_y_start:left_earring_y_end,
                                    left_earring_x_start:left_earring_x_end, c]
                            )
            except ValueError:
                pass

        elif current_position[0]<treshold_right:
            try:
                for c in range(3):
                    image[right_earring_y_start:right_earring_y_end, right_earring_x_start:right_earring_x_end, c] = (
                                    alpha_s_right * resized_right_earring[:, :, c] +
                                    alpha_l_right * image[right_earring_y_start:right_earring_y_end,
                                    right_earring_x_start:right_earring_x_end, c]
                            )
            except ValueError:
                pass

    return image