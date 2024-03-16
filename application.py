from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained CNN model
model = load_model("pretrained_Model/signBWchanged.h5")

# Define labels for hand signs
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize Flask application
app = Flask(__name__)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to generate video feed
def video_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame to extract hand sign
        # Example preprocessing steps from your original code
        cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
        
        cropframe = frame[40:300, 0:300]
        hsv = cv2.cvtColor(cropframe, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        segmented_hand = cv2.bitwise_and(cropframe, cropframe, mask=mask)
        segmented_hand_gray = cv2.cvtColor(segmented_hand, cv2.COLOR_BGR2GRAY)
        segmented_hand_resized = cv2.resize(segmented_hand_gray, (48, 48))
        features = extract_features(segmented_hand_resized)

        # Perform prediction using the pre-trained model
        pred = model.predict(features)
        prediction_label = label[pred.argmax()]

        # Display prediction label on the frame
        accu = "{:.2f}".format(np.max(pred)*100)
        # Get the width of the cropped frame
        frame_width = 300  # Adjust this value according to your actual frame width

        # Draw black rectangle as background for text
        text_bg_color = (0, 0, 0)  # Black color for background
        text = f'{prediction_label}  {accu}%'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x, text_y = 10, 30
        text_bg_width = text_x + text_size[0] + 10  # Add some padding
        cv2.rectangle(cropframe, (text_x, text_y - text_size[1]), (min(text_bg_width, frame_width), text_y), text_bg_color, -1)

        # Write text on top of the black background
        cv2.putText(cropframe, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', cropframe)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route to stream video feed
@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get predicted text
@app.route('/get_prediction')
def get_prediction():
    ret, frame = cap.read()
    if ret:
        cropframe = frame[40:300, 0:300]
        hsv = cv2.cvtColor(cropframe, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        segmented_hand = cv2.bitwise_and(cropframe, cropframe, mask=mask)
        segmented_hand_gray = cv2.cvtColor(segmented_hand, cv2.COLOR_BGR2GRAY)
        segmented_hand_resized = cv2.resize(segmented_hand_gray, (48, 48))
        features = extract_features(segmented_hand_resized)

        # Perform prediction using the pre-trained model
        pred = model.predict(features)
        prediction_label = label[pred.argmax()]
        accuracy = np.max(pred) * 100

        print(f'Prediction: {prediction_label}, Accuracy: {accuracy}')  # Print prediction for debugging

        return jsonify(prediction=prediction_label, accuracy=accuracy)
    else:
        return jsonify(error='Could not read frame')


# Route to render HTML template with video feed
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

# Route to handle stopping the video feed
@app.route('/stop_video')
def stop_video():
    global stream_video
    stream_video = False
    return {'success': True}

# Route to handle restarting the video feed
@app.route('/restart_video')
def restart_video():
    global stream_video
    stream_video = True
    return {'success': True}

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
