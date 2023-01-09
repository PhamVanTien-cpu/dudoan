import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] 
    features = [np.array(int_features)]  
    prediction = model.predict(features)  

    output = prediction[0]
    if(output == 1):
        return render_template('index.html', prediction_text='Bệnh Nhân Có Nguy Cơ Đột Quỵ Cao')
    else:
        return render_template('index.html', prediction_text='Bệnh Nhân Khỏe Mạnh ')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
