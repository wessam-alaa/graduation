import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import mne
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model('model.h5')

scaler = pickle.load(open('scaler.pkl', 'rb'))

class_indices = {0: 'control', 1:'patient'}

def check_patient(results):
    cnt = 0
    for result in results:
        if result > 0.5:
            cnt += 1

    return True if cnt >= 19 else False

#Filtering the data
#write a function that can read the path of these files
def read_data(file_path):
    datax=mne.io.read_raw_edf(file_path,preload=True) #preload should be true for not getting an error
    datax.set_eeg_reference() #by default it will do average of all these channels and consider that average challenge as a reference for other channels
    datax.filter(l_freq=.5 ,h_freq=45) #low frequency -- high frequency  ((We can make high frequency 60))

    #epochs=mne.make_fixed_lenght(data,duration=5,overlap=1) #duration value is in second ---- overlap can be 1 or 2

    epochs=mne.make_fixed_length_epochs(datax,duration=25,overlap=0) #duration value is in second ---- overlap can be 1 or 2
    #the code in up is mne object
    epochs=epochs.get_data()  #to show result of epocs in an array
    return epochs #trials,channel,length

@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['attachment']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        data = read_data(os.path.join('uploads',filename))
        new_array = np.reshape(data, (data.shape[0], 6250, 19))
        scle_data = scaler.transform(new_array.reshape(-1, new_array.shape[-1])).reshape(new_array.shape)
     
        pred = model.predict(scle_data)
        predictions =  class_indices[check_patient(pred)]
        preds = {'prediction':predictions}
        #return render_template('predict.html', predictions=predictions)
        return preds

    return render_template('index.html')

#app.run(host='0.0.0.0', port=80)
app.run()