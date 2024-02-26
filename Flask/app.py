from flask import Flask, render_template,request,jsonify,url_for,redirect
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app=Flask(__name__)
model = tf.keras.models.load_model('Vgg19%.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/input')
def pred():
    return render_template('details.html')

@app.route('/output', methods=['GET','POST'])
def output():
    if request.method =='POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        index=['Bhartanatyam','Kathak','Kathakali','Kuchupudi','Manipuri',
               'Mohiniyattam','Odissi','Sattriya']
        
        img=load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)
        import numpy as np
        x=np.expand_dims(x,axis=0)
        img_data=preprocess_input(x)
        img_data.shape
        output=np.argmax(model.predict(img_data), axis=1)
        result = str(index[output[0]])
        result
        return render_template("result.html", predict = result)

if __name__=='__main__':
    app.run(debug = True,port = 5000)


