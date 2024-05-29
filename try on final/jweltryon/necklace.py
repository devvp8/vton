import cv2
import os
import json

image_directory = 'static/images'

with open('necklace_dimensions.json', 'r') as file:
    necklace_dimensions = json.load(file)
def overlay_necklace(image, face_landmarks,image_collections,selected_indices):
    selected_necklace_index = selected_indices
    selected_necklace_image = image_collections[selected_necklace_index]

    # Calculate the position of the necklace based on face landmarks
    neck_landmark_indices = [152, 5, 11]  # Indices of potential neck landmarks

    # Find the first valid neck landmark from the list
    neck_landmark = None
    for index in neck_landmark_indices:
        if 0 <= index < len(face_landmarks.landmark):
            neck_landmark = face_landmarks.landmark[index]
            break

    if neck_landmark is None:
        return image  # No valid neck landmark found

    # Convert landmark position to image coordinates
    neck_x = int(neck_landmark.x * image.shape[1])
    neck_y = int(neck_landmark.y * image.shape[0])

    # Calculate the position and size of the necklace
    selected_necklace_filename = os.listdir(os.path.join(image_directory, 'necklaces'))[selected_necklace_index]
    necklace_width = necklace_dimensions[selected_necklace_filename]["width"]
    necklace_height = necklace_dimensions[selected_necklace_filename]["height"]

    necklace_x_start = neck_x - int(necklace_width / 2)
    necklace_y_start = neck_y + int(necklace_height / 4)
    necklace_x_end = necklace_x_start + necklace_width
    necklace_y_end = necklace_y_start + necklace_height

    # Resize the necklace image to match the calculated size
    resized_necklace = cv2.resize(selected_necklace_image, (necklace_width, necklace_height))

    # Overlay the resized necklace image on the original image
    alpha_s = resized_necklace[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    try:
        for c in range(3):
            image[necklace_y_start:necklace_y_end, necklace_x_start:necklace_x_end, c] = (
                    alpha_s * resized_necklace[:, :, c] +
                    alpha_l * image[necklace_y_start:necklace_y_end, necklace_x_start:necklace_x_end, c]
            )
    except ValueError:
        pass

    return image