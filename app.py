from flask import Flask, render_template, request, jsonify, redirect, url_for
import face_recognition
import numpy as np
from flask_cors import CORS
from pathlib import Path
import pickle
import cv2
import json
import shutil
import os
import boto3
from botocore.exceptions import NoCredentialsError
# import socket
# import base64
# import io
# from PIL import Image
# import requests


app = Flask(__name__)
CORS(app) 


# ------------------------------------------------------------------------------------------------------------------------------------
# Connecting To AWS

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID_RECOGNITION')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY_RECOGNITION')
AWS_BUCKET = os.getenv('AWS_BUCKET_RECOGNITION')
AWS_LOCATION = os.getenv('AWS_LOCATION_RECOGNITION')

def check_AWS_folder(saved_dir):
    # Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    
    # Check if the object exists in the bucket
    response = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=saved_dir)
    print (response['KeyCount'])

    if response['KeyCount'] != 0:
        return True
    else:
        return False


def connectAWS(saved_dir,saved_folder):

    conn = boto3.Session(AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY)

    s3 = conn.resource('s3',config=boto3.session.Config(connect_timeout=None))
    bucket = s3.Bucket(AWS_BUCKET)
    s3_folder = 'dataset/' + saved_folder
    for root, dirs, files in os.walk(saved_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.relpath(local_path, saved_dir)

            # Check if AWS_LOCATION and s3_folder are valid before joining them
            if AWS_LOCATION and s3_folder:
                s3_key = os.path.join(AWS_LOCATION, s3_folder, s3_path)
            elif AWS_LOCATION:
                s3_key = os.path.join(AWS_LOCATION, s3_path)
            elif s3_folder:
                s3_key = os.path.join(s3_folder, s3_path)
            else:
                s3_key = s3_path

            bucket.upload_file(local_path, s3_key)

def readAWS(saved_dir):

    # Create a Boto3 S3 client
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    # List objects in the specified folder
    response = s3_client.list_objects_v2(Bucket=AWS_BUCKET, Prefix=saved_dir)

    # Check if the folder exists
    if 'Contents' in response:
        for object_data in response['Contents']:
            # Get the key (file path) of each object
            file_path = object_data['Key']

            # Check if the file has the .pkl extension
            if file_path.lower().endswith('.pkl'):
                print("File path:", file_path)

                # Download the .pkl file to the specified local directory
                local_directory = saved_dir # Replace with your desired local directory
                os.makedirs(local_directory, exist_ok=True)
                local_file_path = os.path.join(local_directory, os.path.basename(file_path))
                s3_client.download_file(AWS_BUCKET, file_path, local_file_path)
                print(f"File '{file_path}' downloaded successfully.")
    else:
        print("The folder does not exist in the bucket.")


def connectAWS2(vidpath):

    conn = boto3.Session(AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY)

    s3 = conn.resource('s3')
    awspath = vidpath
    res = s3.Bucket(AWS_BUCKET).upload_file(vidpath,awspath,ExtraArgs={ "ContentType": "video/mp4"})
    print(res)

    if res is None:
        print('File Uploaded Successfully')
    else:
        print('File Not Uploaded')

# Connecting To AWS
# ------------------------------------------------------------------------------------------------------------------------------------





# ------------------------------------------------------------------------------------------------------------------------------------
# Training phase

def video_record():
    # Duration of the video capture (in seconds)
    video_duration = 3

    # Create a video capture object
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam

    # Variables for video recording
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(saved_dir+'/video_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

    # Start video recording
    start_time = cv2.getTickCount()

    while True:
        # Read a frame
        ret, frame = cap.read()

        # Write the frame to the video file
        video_writer.write(frame)

        # Display the frame
        cv2.imshow('Video', frame)

        # Stop video recording after the specified duration
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time >= video_duration:
            break

        # Wait for key press
        key = cv2.waitKey(1)

        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            break

    # Release the video writer object
    video_writer.release()

    # Release the video capture object
    cap.release()
    return "done"


@app.route("/training_recognition",methods=['POST'])
def main():

    nric = request.values['nric']
    name = request.values['name']
    model: str = "hog"
    
    # Directory to store the training images
    global saved_dir
    global saved_folder
    saved_folder = nric + '-' + name + '/'
    saved_dir = 'dataset/' + saved_folder
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    else:
        pass
    DEFAULT_ENCODINGS_PATH = Path(saved_dir + name + ".pkl")
    encodings_location: Path = DEFAULT_ENCODINGS_PATH
   
    # video_record()
    
    data = request.files.get('video')
    vidpath = saved_dir + data.filename
    data.save(vidpath)
    # Open the saved video file
    cap = cv2.VideoCapture(vidpath)

    # Initialize variables
    face_images = []
    names = []
    names2 = []
    encodings = []
    frame_count = 0
    
    # Read frames from the video
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame from BGR to RGB format (required by face_recognition library)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations in the frame
        face_locations = face_recognition.face_locations(rgb_frame,model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)

        for (top, right, bottom, left) in face_locations:
            # Crop the face region
            face = rgb_frame[top:bottom, left:right]

            # Convert the face image from RGB to grayscale
            gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

            # Resize the face image to a fixed size (e.g., 100x100)
            face_resized = cv2.resize(gray_face, (100, 100))
            # Add the face image and name to the lists
            face_images.append(face_resized)
            names.append(name)

            # Save the face image to the training directory
            filename = f'face_{frame_count}.jpg'
            cv2.imwrite(os.path.join(saved_dir, filename), face_resized)
            frame_count += 1

        for encoding in face_encodings:
            names2.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names2, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

    # Release the video capture object
    cap.release()
    data = {"message": True}

    json_data = json.dumps(data)
    connectAWS(saved_dir,saved_folder)
    shutil.rmtree(saved_dir)
    return (json_data)

# Training Phase
# ------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------------
# Recognition Phase

@app.route('/live_recognition')
def index():
    return render_template('index.html')
    
@app.route('/success')
def success():
    return 'Successful'

@app.route('/failed')
def failed():
    return 'Failed'

@app.route('/upload_video',methods = ["POST"])
def upload_video():

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    matches_count = 0
    count = 0
    result_json = {}
    video_loop = True

    name = request.form.get('name')
    nric = request.form.get('nric')
    video_file = request.files['video']
    directory = 'dataset/'
    user =  nric + '-' + name + '/'
    saved_dir = 'dataset/' + nric + '-' + name + '/'
    readAWS(saved_dir)
        
    result = {
        'name': name,
        'nric' : nric,
        'encodings_match': False
        
    }
    
    # Check if video is available
    while video_loop:
        if 'video' not in request.files:
            return ('No video file found in the request.', 400)
        
        # Check if the directory exist or not and make new if not exist
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        else:
            pass

        DEFAULT_ENCODINGS_PATH = Path(saved_dir + name + ".pkl")
        encodings_location: Path = DEFAULT_ENCODINGS_PATH

        # Load Encodings file
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        # Create arrays of known face encodings and their names
        known_face_encodings =  loaded_encodings['encodings']
        known_face_names = loaded_encodings['names']

        # Save the video on the server (adjust the path accordingly).
        vidpath = saved_dir + video_file.filename
        video_file.save(vidpath)

        cap = cv2.VideoCapture(vidpath)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:  # If there are no more frames to read, break out of the loop
                break

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            #  Skip if 4 frames have been checked
            if count <= 3:
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
                    name = "Unknown"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        result['encodings_match'] = True
                        matches_count += 1
                        
                        face_names.append(name)
                count+=1
                print (count)
            else:
                break

        cap.release()

        if matches_count >= 2:
            result['encodings_match'] = True
            result_json = json.dumps(result)
            print(result_json)
            print("success")
            break
        else:
            result['encodings_match'] = False
            result_json = json.dumps(result)
            print(result_json)
            print("failed")
            break
    
    shutil.rmtree(saved_dir)
    return (result_json)

# Recognition Phase
# ------------------------------------------------------------------------------------------------------------------------------------




# Checking if exist
# -----------------------------------------------------------------------------------------------------------------------------------
@app.route('/checking',methods = ["POST"])
def checking():
    nric = request.values['nric']
    name = request.values['name']
    saved_dir = 'dataset/' + nric + '-' + name + '/'
    check_result = check_AWS_folder(saved_dir)
    if check_result:
        result = {'message': True}
    else:
        result = {'message' :False }

    print(check_result)
    return result

        


if __name__ == '__main__':
    envport = (os.getenv('PORT_RECOGNITION'))
    app.run(host='0.0.0.0', port=envport,debug=True) # remove "debug = true" when in production
