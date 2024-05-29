import cv2

def get_landmark_position(face_landmarks, landmark_index):
    landmark = face_landmarks.landmark[landmark_index]
    return landmark.x

def overlay_nosepin(image, face_landmarks,image_collections,selected_indices):
    selected_nosepin_index = selected_indices
    selected_nosepin_image = image_collections[selected_nosepin_index]

    treshold_nose = 0.4280969202518463

    position = get_landmark_position(face_landmarks, 0)
    print(position)

    # Extract the nose coordinates
    nose_landmark = face_landmarks.landmark[5]
    nose_x = int(nose_landmark.x * image.shape[1]) + 20
    nose_y = int(nose_landmark.y * image.shape[0])

    # Calculate the size of the nosepin based on the distance between the eyes
    left_eye_landmark = face_landmarks.landmark[33]
    right_eye_landmark = face_landmarks.landmark[133]
    eye_distance = abs(left_eye_landmark.x - right_eye_landmark.x) * image.shape[1]
    nosepin_size = int(1.5 * eye_distance)

    # Resize the nosepin image
    resized_nosepin = cv2.resize(selected_nosepin_image, (nosepin_size, nosepin_size))

    # Calculate the position to place the nosepin
    top_left = (nose_x - nosepin_size // 2, nose_y - nosepin_size // 2)
    # bottom_right = (nose_x + nosepin_size // 2, nose_y + nosepin_size // 2)
    if position<treshold_nose:
    # Overlay the nosepin onto the frame
        for i in range(nosepin_size):
            for j in range(nosepin_size):
                if resized_nosepin[i, j, 3] > 0:
                    overlay_pixel = resized_nosepin[i, j, :3]
                    alpha = resized_nosepin[i, j, 3] / 255.0
                    image[top_left[1] + i, top_left[0] + j, :] = (1 - alpha) * image[top_left[1] + i, top_left[0] + j, :] + alpha * overlay_pixel

    return image