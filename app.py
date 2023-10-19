from flask import Flask, flash, request, redirect, render_template, url_for, session
from sklearn.preprocessing import StandardScaler
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime,date,time
from keras.preprocessing import image
from keras.metrics import AUC
import pyrebase
from config import firebase_config
import sqlite3 as sql


app = Flask(__name__)
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

dependencies = {"auc_roc": AUC}

verbose_name = {
    0: "Non Demented",
    1: "Very Mild Demented",
    2: "Mild Demented",
    3: "Moderate Demented",
}

# Select model for alzhiemer's
alz_model = load_model("models/alzheimer_cnn_model.h5", compile=False)
alz_model.make_predict_function()
app.config['desktop_path'] = 'static/dataAlzheimers/testsAlzheimers'

 # Select model for breast cancer 
cancer_model = pickle.load(open('models/model.pkl', 'rb'))

# Loading Models
model = joblib.load('ml_model_diabetes')
heartDiseaseModel = joblib.load('ml_model_heart_disease')
modelkidney = joblib.load('Kidney.pkl')
# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"
# A must at all times
app.config['TEMPLATES_AUTO_RELOAD'] = True
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
######################Covid#######
@app.route('/index.html') ##homepageOfcovid
def Index():
    return render_template('index.html')
@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/news.html')
def news():
   return render_template('news.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/faqs.html')
def faqs():
   return render_template('faqs.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')



############################################# BRAIN TUMOR FUNCTIONS ################################################

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/covid')##detection
def covid():
    return render_template('covid.html')
 
@app.route('/upload_chest.html')
def upload_chest():
   return render_template('results_chest.html')


@app.route('/upload_ct.html')
def upload_ct():
   return render_template('results_ct.html')
 
########################### Routing Functions of braintumor ########################################
@app.route('/homebrain')
def main():
    return render_template('homebrain.html')

@app.route('/brain_tumor')
def brain():
    return render_template('brain_tumor.html')

@app.route('/projectBrain')
def projectBrain():
    return render_template('Implementationproject.html')

@app.route('/FAQSBrain')
def faq():
    return render_template('FAQbrain.html')

########################### end Routing Functions of braintumor ########################################

###########################  Function of Alzhiemer's#############################################

def predict_probabilities(img_path):
    test_image = Image.open(img_path).convert("L")
    test_image = test_image.resize((128, 128))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(-1, 128, 128, 1)

    predict_probabilities = alz_model.predict(test_image)

    class_probabilities = [probability * 100 for probability in predict_probabilities[0]]

    return class_probabilities

################################### Routing Functions of Alzhiemer's#######################################

@app.route("/alzhiemerhome", methods=["GET", "POST"])
def alzhiemerhome():
    return render_template("indexAlzheimers.html")


@app.route("/submit", methods=["POST"])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]

        if img:
            img.save(os.path.join(app.config['desktop_path'], img.filename))
            img_path = os.path.join(app.config['desktop_path'], img.filename)
            class_probabilities = predict_probabilities(img_path)
        else:
            # Handle the case where no image was uploaded
            img_path = ""
            class_probabilities = None

    return render_template(
        "classifier.html", class_probabilities=class_probabilities, img_path=img_path
    )

@app.route("/previous-results", methods=["GET"])
def previous_results():
    return render_template("previous_results.html")


@app.route("/faqsAlzhiemer", methods=["GET"])
def faqsAlzhiemer():
    return render_template("FAQsAlzheimers.html")

@app.route("/classifier", methods=["GET"])
def classifier():
    return render_template("classifier.html")

@app.route("/aboutAlzhiemer", methods=["GET"])
def aboutAlzhiemer():
    # Render the About.html page
    return render_template("AboutAlzheimers.html")

@app.route("/game", methods=["GET"])
def game():
    # Render the About.html page
    return render_template("memory_game.html")
 
 ######################## end rounting functions #######################################################

 ########################### breast cancer function ###################################################
@app.route('/predict',methods=['POST'])
def predict_cancer():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses']

  df = pd.DataFrame(features_value, columns=features_name)
  output = cancer_model.predict(df)
  if output == 4:
      res_val = "Breast cancer"
  else:
      res_val = "no Breast cancer"

  return render_template('cancer_detection.html', prediction_text='Patient has {}'.format(res_val))

 ###########################breast cancer webApp###########################################
@app.route('/index_cancer.html')
def index_cancer():
   return render_template('index_cancer.html')

@app.route('/menu-bar-charity.html')
def menu_bar_charity():
   return render_template('menu-bar-charity.html')

@app.route('/footer.html')
def footer():
   return render_template('footer.html')

@app.route('/our-causes.html')
def our_causes():
   return render_template('our-causes.html')

@app.route('/about-us.html')
def about_us():
   return render_template('about-us.html')

@app.route('/cancer_detection.html')
def cancer_detection():
   return render_template('cancer_detection.html')

@app.route('/Analyzer.html')
def analyzer():
   return render_template('Analyzer.html')

########################### end breast cancer routing functions ###################################

########################### Result Functions ########################################

######covid
@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   resnet_chest = load_model('jupyterNotebooksCOVID/resnet_chest.h5')
   vgg_chest = load_model('jupyterNotebooksCOVID/vgg_chest.h5')
   inception_chest = load_model('jupyterNotebooksCOVID/inceptionv3_chest.h5')
   xception_chest = load_model('jupyterNotebooksCOVID/xception_chest.h5')

   image = cv2.imread('./static/uploads/assets/images/upload_chest.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   resnet_pred = resnet_chest.predict(image)
   probability = resnet_pred[0]
   print("Resnet Predictions:")
   if probability[0] > 0.5:
      resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(resnet_chest_pred)

   vgg_pred = vgg_chest.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   if probability[0] > 0.5:
      vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(vgg_chest_pred)

   inception_pred = inception_chest.predict(image)
   probability = inception_pred[0]
   print("Inception Predictions:")
   if probability[0] > 0.5:
      inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(inception_chest_pred)

   xception_pred = xception_chest.predict(image)
   probability = xception_pred[0]
   print("Xception Predictions:")
   if probability[0] > 0.5:
      xception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      xception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(xception_chest_pred)

   return render_template('results_chest.html',resnet_chest_pred=resnet_chest_pred,vgg_chest_pred=vgg_chest_pred,inception_chest_pred=inception_chest_pred,xception_chest_pred=xception_chest_pred)
##ct
@app.route('/uploaded_ct', methods = ['POST', 'GET'])
def uploaded_ct():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_ct.jpg'))

   resnet_ct = load_model('jupyterNotebooksCOVID/resnet_ct.h5')
   vgg_ct = load_model('jupyterNotebooksCOVID/vgg_ct.h5')
   inception_ct = load_model('jupyterNotebooksCOVID/inception_ct.h5')
   xception_ct = load_model('jupyterNotebooksCOVID/xception_ct.h5')

   image = cv2.imread('./static/uploads/assets/images/upload_ct.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   resnet_pred = resnet_ct.predict(image)
   probability = resnet_pred[0]
   print("Resnet Predictions:")
   if probability[0] > 0.5:
      resnet_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      resnet_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(resnet_ct_pred)

   vgg_pred = vgg_ct.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   if probability[0] > 0.5:
      vgg_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      vgg_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(vgg_ct_pred)

   inception_pred = inception_ct.predict(image)
   probability = inception_pred[0]
   print("Inception Predictions:")
   if probability[0] > 0.5:
      inception_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      inception_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(inception_ct_pred)

   xception_pred = xception_ct.predict(image)
   probability = xception_pred[0]
   print("Xception Predictions:")
   if probability[0] > 0.5:
      xception_ct_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   else:
      xception_ct_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
   print(xception_ct_pred)

   return render_template('results_ct.html',resnet_ct_pred=resnet_ct_pred,vgg_ct_pred=vgg_ct_pred,inception_ct_pred=inception_ct_pred,xception_ct_pred=xception_ct_pred)



####### brain #########################
def predict_Btumor(img_path):
    model_load = load_model("Trained ModelBrain/brain_tumor.h5")

    img = cv2.imread(img_path)
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    preds = model_load.predict(img)
    return preds[0]

@app.route('/predictBTumor', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploadsbrain', secure_filename(f.filename))
        print(file_path)
        preds = predict_Btumor(file_path)
        print(preds)

        if int(preds[0]) == 0:
            result = "No worry! No Brain Tumor"
        else:
            result = "Patient has Brain Tumor"

        print(f'prdicted: {result}')

        return result

    return None
# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route("/indexdiabetes.html")
def pageOfdiabetes():	
	return render_template('indexdiabetes.html') 

@app.route("/Implementaiondiabetes.html")
def Implementdiabetes():	
	return render_template('Implementaiondiabetes.html')

@app.route("/diabetesdetection")
def diabetesDetection():	
	return render_template('diabetes.html')

@app.route("/heartdisease-detection")
def heartDiseaseDetection():
	return render_template('heart.html')

@app.route("/contact_diabetes.html")
def contactDiabetes():	
	return render_template('contact_diabetes.html')

@app.route("/treatment-and-medication.html")
def medication():	
	return render_template('treatment-and-medication.html')

@app.route("/types-of-diabetes.html")
def types():	
	return render_template('types-of-diabetes.html')

@app.route("/signs-and-symptoms.html")
def symptoms():	
	return render_template('signs-and-symptoms.html')

@app.route("/physical-activity.html")
def physical():	
	return render_template('physical-activity.html')

@app.route("/diet-and-nutrition.html")
def diet():	
	return render_template('diet-and-nutrition.html')

@app.route("/diabetes-risks.html")
def risks():	
	return render_template('diabetes-risks.html')

@app.route("/diabetes-hypertension-link.html")
def hypertension():	
	return render_template('diabetes-hypertension-link.html')

@app.route("/calorie-calculator.html")
def calorie():	
	return render_template('calorie-calculator.html')

@app.route("/bmi-calculator.html")
def bmi():	
	return render_template('bmi-calculator.html')

@app.route("/KidneyDisease")
def KidneyDisease():	
	return render_template('indexkidney.html')


@app.route("/diabetesdetection/predictdiabetes", methods=['POST', 'GET'])
def showDiabetesResult():
    scaler = StandardScaler()

    numOfPreg = request.form['a']
    glucose = request.form['b']
    bloodPressure = request.form['c']
    skinThickness = request.form['d']
    insulin = request.form['e']
    bmi = request.form['f']
    diabPedigFunc = request.form['g']
    age = request.form['h']

    arr = np.array([[numOfPreg, glucose, bloodPressure, skinThickness, insulin, bmi, diabPedigFunc, age]], dtype=float)
    
    # Assuming "model" is your trained machine learning model
    pred = model.predict(arr)  # This gives a binary 0 or 1 prediction

    # For probability, you can use model.predict_proba if it's a classifier that supports probabilities
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(arr)[:, 1]  # Assuming it's a binary classifier
    else:
        # If the model doesn't support predict_proba, you can use a different method

        # For instance, if it's a logistic regression model
        log_reg_decision = model.decision_function(arr)
        probability = 1 / (1 + np.exp(-log_reg_decision))

    if pred[0] == 1:
        result = "HAS DIABETES"
        probability_percentage = int(probability * 100)
    else:
        result = "SAFE"
        probability_percentage = 0  # Set probability to 0 when the person is safe

    return render_template('diabetesResult.html', data=result, probability=probability_percentage)


@app.route("/heartdisease-detection/predictHeartDisease", methods=['POST', 'GET'])
def showHeartDiseaseResult():
	age = request.form['age']
	sex = request.form['sex']
	chestPainType = request.form['chestPainType']
	restingBloodPressure = request.form['restingBloodPressure']
	cholesterol = request.form['cholesterol']
	fastingBloodSugar = request.form['fastingBloodSugar']
	restingElectroResult = request.form['restingElectroResult']
	maxHeartRate = request.form['maxHeartRate']
	exInducedAngina = request.form['exInducedAngina']
	oldPeakDep = request.form['oldPeakDep']
	slope = request.form['slope']
	ca = request.form['ca']
	thaliumStress = request.form['thaliumStress']

	heartArr = np.array([[age, sex, chestPainType, restingBloodPressure, cholesterol, 
		fastingBloodSugar, restingElectroResult, maxHeartRate, exInducedAngina, 
		oldPeakDep, slope,ca, thaliumStress]], dtype=float)

	heartPred = heartDiseaseModel.predict(heartArr)

	if heartPred[0] == 1:
		result = "HAS A HEART DISEASE"

	else:
		result = "SAFE"


	return render_template('heartResult.html', data=result)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction = modelkidney.predict(values)

        return render_template('kidneyresult.html', prediction=prediction)





if __name__ == '__main__':
    app.run(debug=True)
