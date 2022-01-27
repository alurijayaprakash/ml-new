import os
from flask import Flask, render_template, request, redirect, url_for, abort, flash
from werkzeug.utils import secure_filename
import pandas as pd
from lr_model import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.txt', '.csv']
app.config['UPLOAD_PATH'] = 'uploads'
app.secret_key = "abc"

# @app.errorhandler(400)
# def too_large(e):
#     # return "File Extension not supported", 400
#     return render_template('400.html'), 400



@app.route('/')
def index():
    print("Excecurted.....")
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():

    print("Excecurted_____upload")
    uploaded_file = request.files['file']
    has_index = "know_index" in request.form
    isExpert = "know_user" in request.form
    filename = secure_filename(uploaded_file.filename)
    print("My File Name : ", filename, has_index, isExpert)
    if filename == '':
            flash('No selected file')
            return redirect(request.url)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]

        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            flash('File Extension not supported, Please upload TXT, CSV files only')
            return render_template('index.html')
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        flash('Your file accepted by this ML tool box')
        if file_ext == ".txt":
            print(os.path.join(app.config['UPLOAD_PATH'])+"\\"+filename)
            mypath = os.path.join(app.config['UPLOAD_PATH'])+"\\"+filename
            # mydata = np.loadtxt(mypath, delimiter=',', dtype=float)
            mydata = pd.read_csv(mypath, sep = ',')
        elif file_ext == ".csv":
            mydata = pd.read_csv(os.path.join(app.config['UPLOAD_PATH'], filename))
        else:
            return "File not supported by this ML Toolbox"
        if has_index:
            mydata = mydata.iloc[: , 1:]
        print(sample())
        print(mydata.head())
        output = lr_model_alog(mydata)
        # output = 1
        print(output)
    return render_template('index.html', data = output)

def upload_files1():
    flash('XXXXXXXXXXXXXXX')
    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True)

## For deploying the app use `app.run(debug=False, host="0.0.0.0", port=80)`