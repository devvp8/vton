import cv2
import numpy as np
import mediapipe as mp

def get_hand_landmark_position(hand_landmarks, landmark_index):
    landmark = hand_landmarks.landmark[landmark_index]
    return landmark.x,landmark.y

def overlay_braclet(image, hand_landmarks,image_collections,selected_indices):
    selected_braclet_index = selected_indices
    selected_braclet_image = image_collections[selected_braclet_index]
    wrist_landmark_index = 0

    # for out of bound-->overlay stopped.
    treshold_x=0.16892462968826294
    treshold_y=0.1761983036994934

    hand_landmark_20_x,hand_landmark_20_y = get_hand_landmark_position(hand_landmarks, 20)
    hand_landmark_4_x,hand_landmark_4_y = get_hand_landmark_position(hand_landmarks, 4)
    hand_landmark_5_x, hand_landmark_5_y = get_hand_landmark_position(hand_landmarks, 5)
    hand_landmark_17_x, hand_landmark_17_y = get_hand_landmark_position(hand_landmarks, 17)

    distance_x = np.linalg.norm(np.array(hand_landmark_5_x) - np.array(hand_landmark_17_x))
    distance_y = np.linalg.norm(np.array(hand_landmark_5_y) - np.array(hand_landmark_17_y))

    # Get wrist landmark coordinates
    wrist_x = int(hand_landmarks.landmark[wrist_landmark_index].x * image.shape[1])
    wrist_y = int(hand_landmarks.landmark[wrist_landmark_index].y * image.shape[0])

    # Calculate bracelet size .28
    bracelet_height = int(image.shape[0] * 0.34)
    bracelet_width = int(bracelet_height * (selected_braclet_image.shape[1] / selected_braclet_image.shape[0]))

    bracelet_x_start = wrist_x - int(bracelet_width / 2)
    bracelet_y_start = wrist_y + int(bracelet_height / 4) - 30

    if distance_x>treshold_x or distance_y>treshold_y:
    # Calculate bracelet position based on hand orientation
        if hand_landmark_20_x < hand_landmark_4_x:  # right hand
            if hand_landmarks.landmark[0].x > hand_landmarks.landmark[5].x:
                bracelet_y_start = wrist_y - int(bracelet_width / 2) + 13
                selected_braclet_image = cv2.rotate(selected_braclet_image, cv2.ROTATE_90_CLOCKWISE)

        if hand_landmark_20_x > hand_landmark_4_x:  # left hand
            if hand_landmarks.landmark[0].x < hand_landmarks.landmark[5].x:  # left hand
                bracelet_y_start = wrist_y - int(bracelet_width / 2) + 6
                selected_braclet_image = cv2.rotate(selected_braclet_image, cv2.ROTATE_90_CLOCKWISE)

        bracelet_x_end = bracelet_x_start + bracelet_width
        bracelet_y_end = bracelet_y_start + bracelet_height

        # Resize bracelet image
        resized_bracelet = cv2.resize(selected_braclet_image, (bracelet_width, bracelet_height))

        # Overlay the resized braclet image on the original image
        alpha_s = resized_bracelet[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        try:
                # Overlay bracelet on the image
            for c in range(3):
                    image[bracelet_y_start:bracelet_y_end, bracelet_x_start:bracelet_x_end, c] = (alpha_s * resized_bracelet[:, :, c] +
                    alpha_l * image[bracelet_y_start:bracelet_y_end, bracelet_x_start:bracelet_x_end, c])

        except ValueError:
                pass

    return image