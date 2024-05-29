import cv2
import numpy as np
import mediapipe as mp

def get_hand_landmark_position(hand_landmarks, landmark_index):
    landmark = hand_landmarks.landmark[landmark_index]
    return landmark.x,landmark.y

#main overlay function
def overlay_ring(image, hand_landmarks, image_collections, selected_indices):
    selected_ring_index = selected_indices
    selected_ring_image = image_collections[selected_ring_index]

    treshold_x = 0.16892462968826294
    treshold_y = 0.1761983036994934

    hand_landmark_5_x, hand_landmark_5_y = get_hand_landmark_position(hand_landmarks, 5)
    hand_landmark_17_x, hand_landmark_17_y = get_hand_landmark_position(hand_landmarks, 17)

    distance_x = np.linalg.norm(np.array(hand_landmark_5_x) - np.array(hand_landmark_17_x))
    distance_y = np.linalg.norm(np.array(hand_landmark_5_y) - np.array(hand_landmark_17_y))

    # Extract the coordinates of the ring finger tip
    ring_finger_tip = hand_landmarks.landmark[13]

    image_height, image_width, _ = image.shape

    #for detection of hand-left or right
    #considering pinky finger and thumb for accurate results
    hand_landmark_20_x = get_hand_landmark_position(hand_landmarks, 20)
    hand_landmark_4_x = get_hand_landmark_position(hand_landmarks, 4)

        # Convert the ring finger tip coordinates to pixel values
    ring_finger_tip_x = int(ring_finger_tip.x * image_width) + 14
    ring_finger_tip_y = int(ring_finger_tip.y * image_height) - 24

        # Calculate the size of the ring based on the distance between wrist and finger tip
    wrist_landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    wrist_x = int(wrist_landmark.x * image_width)
    wrist_y = int(wrist_landmark.y * image_height)
    distance = np.linalg.norm(np.array([wrist_x, wrist_y]) - np.array([ring_finger_tip_x, ring_finger_tip_y]))
    ring_size = int(distance * 0.6)  # Adjust the scaling factor as needed

        # Resize the ring image to the calculated size
    resized_ring = cv2.resize(selected_ring_image, (ring_size, ring_size))

    if distance_x > treshold_x or distance_y > treshold_y:
        #condition for rotating ring acc to hand
        if hand_landmark_20_x < hand_landmark_4_x: #right hand
            if hand_landmarks.landmark[0].x > hand_landmarks.landmark[5].x:
                    ring_finger_tip_x = ring_finger_tip_x - 30 #adjust ring position
                    ring_finger_tip_y = ring_finger_tip_y + 10
                    resized_ring = cv2.rotate(resized_ring, cv2.ROTATE_90_CLOCKWISE)

            #condition for rotating ring acc to hand
        if hand_landmark_20_x > hand_landmark_4_x: #left hand
            if hand_landmarks.landmark[0].x < hand_landmarks.landmark[5].x: #left hand
                    ring_finger_tip_x = ring_finger_tip_x + 33 #adjust ring position
                    ring_finger_tip_y = ring_finger_tip_y + 10
                    resized_ring = cv2.rotate(resized_ring, cv2.ROTATE_90_CLOCKWISE)

            # Calculate the position of the ring on the hand
        ring_x = ring_finger_tip_x - int(ring_size / 2) - 20
        ring_y = ring_finger_tip_y - int(ring_size / 2)

            # Adjust the position of the ring to prevent it from going out of bounds
        if ring_x < 0:
            ring_x = 0
        if ring_y < 0:
            ring_y = 0
        if ring_x + ring_size > image_width:
            ring_x = image_width - ring_size
        if ring_y + ring_size > image_height:
            ring_y = image_height - ring_size

            # Extract the alpha channel from the ring image
        alpha = resized_ring[:, :, 3] / 255.0

            # Extract the RGB channels from the ring image
        ring_rgb = resized_ring[:, :, :3]

            # Resize the overlay to match the size of the region of interest (ROI)
        resized_overlay = cv2.resize(ring_rgb, (ring_size, ring_size))

            # Multiply the resized overlay with the alpha channel
        overlay = (resized_overlay * alpha[:, :, np.newaxis]).astype(np.uint8)

            # Add the overlay to the ROI in the result frame
        image[ring_y:ring_y + ring_size, ring_x:ring_x + ring_size] = cv2.add(overlay, image[ring_y:ring_y + ring_size, ring_x:ring_x + ring_size])

    return image