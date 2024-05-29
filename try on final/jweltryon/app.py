from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import os
import openai
import numpy as np
import builtins
import dlib
import math
import nltk
from nltk.stem import WordNetLemmatizer
import imutils
from imutils import face_utils
from jweltryon.earring import overlay_earrings
from jweltryon.necklace import overlay_necklace
from jweltryon.braclet import overlay_braclet
from jweltryon.ring import overlay_ring
from jweltryon.nosepin import overlay_nosepin

app = Flask(__name__)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands=mp.solutions.hands
openai.api_key = os.getenv('OPENAI_API_KEY')
nltk.download('wordnet')

# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.8)
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

shape_predictor_path="C:\\Users\\Dev Atul Patel\\OneDrive\\Desktop\\Machine Learning\\shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Specify the directory where the images are located
image_directory = 'static/images'

jewelry_types = ['earring', 'necklace', 'nosepin', 'ring', 'braclet']
display_status = {jewelry_type: False for jewelry_type in jewelry_types}
selected_indices = {jewelry_type: None for jewelry_type in jewelry_types}
finding_shape= False
detected_face=""

def get_landmark_position(face_landmarks, landmark_index):
    landmark = face_landmarks.landmark[landmark_index]
    return landmark.x

def get_hand_landmark_position(hand_landmarks, landmark_index):
    landmark = hand_landmarks.landmark[landmark_index]
    return landmark.x,landmark.y

subdirectories = ['necklaces', 'nosepin', 'ring', 'braclets']
image_collections = {subdir: [] for subdir in subdirectories}
suitable_earring = []

for subdir in subdirectories:
    subdirectory_path = os.path.join(image_directory, subdir)
    if not os.path.exists(subdirectory_path):
        continue

    for filename in os.listdir(subdirectory_path):
        if filename.endswith('.png'):
            image = cv2.imread(os.path.join(subdirectory_path, filename), cv2.IMREAD_UNCHANGED)
            image_collections[subdir].append(image)


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def select_earrings_for_face_shape(face_shape):
    earrings_folder = os.path.join(image_directory, 'earrings', face_shape)
    for filename in os.listdir(earrings_folder):
        if filename.endswith('.png'):
            image = cv2.imread(os.path.join(earrings_folder, filename), cv2.IMREAD_UNCHANGED)
            print(filename)
            suitable_earring.append(image)

    return suitable_earring


def detect_face_shape(frame):
    global finding_shape,detected_face,suitable_earring
    image = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    count = 0

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        for (x, y) in shape:
            count = count + 1
            if (count == 1):
                (x1, y1) = (x, y)
            if (count == 3):
                (x3, y3) = (x, y)
            if (count == 5):
                (x5, y5) = (x, y)
            if (count == 7):
                (x7, y7) = (x, y)
            if (count == 9):
                (x9, y9) = (x, y)
            if (count == 17):
                (x17, y17) = (x, y)
            if (count == 28):
                (x28, y28) = (x, y)

        slope1 = ((y3 - y1) * (1.0)) / ((x3 - x1) * (1.0))
        slope3 = ((y7 - y5) * (1.0)) / ((x7 - x5) * (1.0))

        distx = math.sqrt(pow((x1 - x17), 2) + pow((y1 - y17), 2))
        disty = math.sqrt(pow((x9 - x28), 2) + pow((y9 - y28), 2))
        thresh = distx - disty

        lg = builtins.open("long.txt", 'r')
        rnd = builtins.open("round.txt", 'r')
        het = builtins.open("heart.txt", 'r')
        squ = builtins.open("square.txt", 'r')

        thresh_lg = float(lg.readline())
        thresh_rnd = float(rnd.readline())
        thresh_het = float(het.readline())
        thresh_squ = float(squ.readline())

        lg.seek(0)
        rnd.seek(0)
        het.seek(0)
        squ.seek(0)
        avg_hr = (thresh_rnd + thresh_het) / (2.0)
        avg_ls = (thresh_lg + thresh_squ) / (2.0)

        total_thresh = (avg_hr + avg_ls) / (2.0)

        print("+++++++++++++++")
        if thresh <= total_thresh:
            # print("long or square")
            if slope1 >= 7.395:
                if slope3 >= 1.15:
                    detected_face = "long"
                else:
                    detected_face = "square"

            elif slope1 < 7.395:
                if slope3 >= 1.15:
                    detected_face = "square"

                else:
                    detected_face = "long"

        if thresh > total_thresh:
            # print("round or heart")
            if slope1 >= 11.75:
                if slope3 <= 1.1:
                    detected_face = "heart"
                else:
                    detected_face = "round"

            elif slope1 < 11.75:
                if slope3 > 1.1:
                    detected_face = "round"
                else:
                    detected_face = "heart"

    print("Detected face shape:", detected_face)
    suitable_earring = []
    select_earrings_for_face_shape(detected_face)
    finding_shape = True

def generate_frames():
    open_capture()
    global display_status, selected_indices,finding_shape,detected_face

    ret, first_frame = cap.read()
    first_frame = cv2.flip(first_frame,1)

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    if not ret:
        exit()

    while True:

        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        if not ret:
            break

        if not finding_shape:
            detect_face_shape(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x,y,w,h) in faces:
            roi_first_frame = first_frame_rgb[y:y+h, x:x+w]
            roi_frame = rgb_frame[y:y+h, x:x+w]
            diff = cv2.absdiff(roi_frame,roi_first_frame)
            if np.mean(diff) > 70:
                print("new detected")
                detect_face_shape(frame)
        #
        # results = face_mesh.process(rgb_frame)
        # results_hand = hands.process(rgb_frame)

        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         if display_status['earring']:
        #             display_status['ring'] = False
        #             display_status['braclet'] = False
        #             frame = overlay_earrings(frame, face_landmarks, suitable_earring, selected_indices['earring'])
        #
        #         if display_status['necklace']:
        #             display_status['ring'] = False
        #             display_status['braclet'] = False
        #             frame = overlay_necklace(frame, face_landmarks, image_collections['necklaces'], selected_indices['necklace'])
        #
        #         if display_status['nosepin']:
        #             display_status['ring'] = False
        #             display_status['braclet'] = False
        #             frame = overlay_nosepin(frame, face_landmarks, image_collections['nosepin'], selected_indices['nosepin'])
        #
        # if results_hand.multi_hand_landmarks:
        #     for hand_landmarks in results_hand.multi_hand_landmarks:
        #         if display_status['ring']:
        #             display_status['earring'] = False
        #             display_status['necklace'] = False
        #             display_status['nosepin'] = False
        #             frame = overlay_ring(frame, hand_landmarks, image_collections['ring'], selected_indices['ring'])
        #
        #         if display_status['braclet'] :
        #             display_status['earring'] = False
        #             display_status['necklace'] = False
        #             display_status['nosepin'] = False
        #             frame = overlay_braclet(frame, hand_landmarks, image_collections['braclets'], selected_indices['braclet'])

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


def open_capture():
    global cap
    cap = cv2.VideoCapture(0)

def close_capture():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None

# Create the face mesh model
@app.route('/open')
def open():
    open_capture()
    return 'Video capture opened'
@app.route('/close')
def close():
    close_capture()
    return 'Video capture closed'
@app.route('/')
def home():
    global finding_shape,suitable_earring,display_status,selected_indices,detected_face
    finding_shape = False
    detected_face = ''
    suitable_earring = []
    display_status = {jewelry_type: False for jewelry_type in jewelry_types}
    selected_indices = {jewelry_type: None for jewelry_type in jewelry_types}
    return render_template('index.html')


@app.route('/speech', methods=['POST'])
def speech():
    global selected_indices
    spoken = request.form['spokenText']

    prompt = f"""
        Translate the following text to English:
        ```
        {spoken}
        ```
        """

    response = get_completion(prompt)
    print(response)

    selected_indices = {
        'necklace': 1,
        'nosepin' : 1,
        'ring' : 1,
        'braclet' : 1,
        'earring' : 1
    }
    response = response.lower()
    response = response.replace('.', '')
    newtext = response.split(" ")
    lemmatizer = WordNetLemmatizer()
    newtext_lemmatized = [lemmatizer.lemmatize(word) for word in newtext]

    print(newtext_lemmatized)

    for new in newtext_lemmatized:
        if new in jewelry_types:
            jewelry_type = new
            display_status[jewelry_type] = not display_status[jewelry_type]
    return 'success'

@app.route('/result', methods=['GET', 'POST'])
def result():
    global display_status, selected_indices, detected_face

    if request.method == 'POST':
        for jewelry_type in jewelry_types:
            if jewelry_type in request.form:
                display_status[jewelry_type] = not display_status[jewelry_type]

            select_key = f'{jewelry_type}_select'
            if select_key in request.form:
                selected_indices[jewelry_type] = int(request.form[select_key])

    earring_options = [{'index': i, 'filename': f'Earring {i + 1}'} for i in range(len(suitable_earring))]
    necklace_options = [{'index': i, 'filename': f'Necklace {i + 1}'} for i in
                        range(len(image_collections['necklaces']))]
    nosepin_options = [{'index': i, 'filename': f'Nosepin {i + 1}'} for i in range(len(image_collections['nosepin']))]
    ring_options = [{'index': i, 'filename': f'Ring {i + 1}'} for i in range(len(image_collections['ring']))]
    braclet_options = [{'index': i, 'filename': f'Braclet {i + 1}'} for i in range(len(image_collections['braclets']))]

    return render_template('result.html', jewelry_types=jewelry_types,
                           display_status=display_status, selected_indices=selected_indices,
                           earring_options=earring_options,
                           necklace_options=necklace_options, nosepin_options=nosepin_options,
                           ring_options=ring_options, braclet_options=braclet_options)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)