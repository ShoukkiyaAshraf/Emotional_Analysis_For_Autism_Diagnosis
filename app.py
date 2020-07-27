from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from time import sleep
import cv2
from tkinter import *

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from flask import request
#from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Testing_model/VGG19_result.h5'
face_classifier = cv2.CascadeClassifier('Testing_model/haarcascade_frontalface_default.xml')

# Load your trained model
# load json and create model
json_file = open('Testing_model/VGG19_result.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("Testing_model/VGG19_result.h5")
print("Loaded model")

#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def alert_popup(title, message, path):
    """Generate a pop-up window for special messages."""
    root = Tk()
    root.title(title)
    w = 400     # popup window width
    h = 200     # popup window height
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w)/2
    y = (sh - h)/2
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    m = message
    m += '\n'
    m += path
    w = Label(root, text=m, width=120, height=10)
    w.pack()
    b = Button(root, text="OK", command=root.destroy, width=10)
    b.pack()
    mainloop()

def model_predict(model,filename):
    
    #class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
    # Emotions dictionary
    emotions = {"anger" : 0,"disgust" : 1,"fear" : 2,"happy" : 3,"sad" : 4,"surprise" : 5,"neutral" : 6}
    cap = cv2.VideoCapture(filename)
    file = open('static/report.txt', 'w')
    file.write(' Emotion   time \n')
    while True:
    # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            #roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)


            if np.sum([roi_gray])!=0:
                #roi = roi_gray.astype('float')/255.0
                #roi = img_to_array(roi)
                #roi = np.expand_dims(roi,axis=0)
                # Resize for our model (48x48x1)
                small = cv2.resize(roi_gray, dsize = (48,48))
                # convert size from 48x48 to 1x48x48
                image3D = np.expand_dims(small,axis = 0)
                # convert to 1x48x48x1
                image4D = np.expand_dims(image3D, axis = 3)
                # convert to 1x48x48x3
                image4D3 = np.repeat(image4D, 3, axis=3)

            # make a prediction on the ROI, then lookup the class

                preds = model.predict(image4D3)[0]
                listt = [1 if metric == preds.max() else 0 for metric in preds]
                # Get the index 1 in the binary list, listt 
                emotion_index = listt.index(1)
                emotion = list(emotions.keys())[emotion_index]
                #label=class_labels[preds.argmax()]
                label_position = (x,y)
                fps = cap.get(cv2.CAP_PROP_FPS) 
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                duration = frame_count / fps
                minutes = int(duration / 60)
                seconds = duration % 60
                seconds = round(seconds,1)
                string = ' -  ' +  str(minutes) + ':' + str(seconds)
                print('{}'.format(emotion)+string)
                file.write('\n    ') 
                file.write('{}'.format(emotion)+string)
                cv2.putText(frame,'{}'.format(emotion)+string,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            file.close()
            break
    file.close()
    cap.release()
    cv2.destroyAllWindows()
    return string


@app.route('/')
def index():
    # Main page
    return render_template('login.html')

@app.route('/login',methods=['POST'])
def login():
    #if request.method == 'POST':
    username=request.form.get('userid')
    password = request.form.get('pswrd')
    print(username)
    print(password)
    actual_username = 'user101'
    actual_password = 'password'
    print(type(username))
    if(username == actual_username and password == actual_password):
        print('success')
        return render_template('index.html')
    else :
        return '<h1> HTTP error 404 </h1> <br> <p> You do\'nt have permmission to access this page </p>'

@app.route('/about',methods=['GET'])
def about():
    if request.method == 'GET' :
        return render_template('about.html')
    else:
        return 'error'
@app.route('/upload_file',methods=['GET','POST'])
def upload_file():
    if 'file' not in request.files:
            return "nothing"
    else :
        file = request.files['file']
        return file.filename
            
@app.route('/home',methods=['GET'])
def home():
    if request.method == 'GET' :
        return render_template('index.html')
    else:
        return 'error'
    
@app.route('/logout',methods=['GET'])
def logout():
    if request.method == 'GET' :
        return render_template('login.html')
    else:
        return 'error'

@app.route('/casestudy',methods=['GET'])
def casestudy():
    if request.method == 'GET' :
        return render_template('liam.html')
    else:
        return render_template('index.html')
    
@app.route('/report',methods=['GET'])
def report():
    if request.method == 'GET' :
        f = open('static/report.txt', 'r')
        content = f.read()
        return render_template('report.html',content=content)
    else:
        return render_template('report.html','cannot retrieve report')    

@app.route('/compare',methods=['GET'])
def compare():
    if request.method == 'GET':
        actual_file = open('static/actual.txt','r').readlines() #contains the actual  emotion data of video file
        check_file = open('static/report.txt','r').readlines() #contains the analysed emotion report of child 
        actual_file_line = []
        for lines in actual_file:
            actual_file_line.append(lines)
        check_file_line = []
        for linesc in check_file:
            check_file_line.append(linesc)
        n = 0
        flag=0
        for linea in check_file_line:
            if linea == actual_file_line[n]:
                flag += 1
                n += 1
            else:
                n += 1
        match = round(((flag / len(actual_file_line)) * 100),3)
        match = str(match) + '%'
        return render_template('final_result_ADOS.html',match=match)
    else :
        return "error"
    
    
@app.route('/ados',methods=['GET'])
def ados():
    if request.method == 'GET' :
        return render_template('index_updated.html')
    else:
        return render_template('index.html')        

@app.route('/predict', methods=['POST'])
def predict():
    #render_template('index_updated.html')
    checky = "not available"
    if 'file' not in request.files:
            return "nothing"
    else :
        file = request.files['file']
    if request.method == 'POST':
        loading = "loading....."
        #render_template('index_updated.html',loading=loading)
        output = model_predict(model,file.filename)
        check = "available here "
        alert_popup("Success!", "Processing completed. Your report was saved as ", "report.txt" )
        return render_template('index_updated.html',output1=check)
    return ender_template('index_updated.html',output1=checky)


if __name__ == '__main__':
    app.run(debug=True)