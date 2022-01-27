import os
from flask import Flask, render_template, request, redirect, url_for, abort, flash
from werkzeug.utils import secure_filename
import pandas as pd
from lr_normal import *
from lr_expert import *
from lr_normal_multi import *
from lr_expert_multi import *
from lg_normal import *
from lg_multi_expert import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10240 * 10240
app.config['UPLOAD_EXTENSIONS'] = ['.txt', '.csv']
app.config['UPLOAD_PATH'] = 'uploads'
app.secret_key = "abc"



# Index Page -----------------------------------------------------------------------------------

@app.route('/')
def index():
    print("Index page rendered.....")
    return render_template('index.html')

# Index Page -----------------------------------------------------------------------------------

@app.route('/', methods=['POST'])
def readmode():
    print("Index page need to select Mode")
    # selected_data = "know_user" in request.form
    # selected_data = request.form.get('customSwitch1')
    global isExpertType
    global algotype
    option = request.form.getlist('modeof')[0]
    algotype = request.form.getlist('algotype')[0]
    print("option & algotype ===> ", option, algotype)
    
    if algotype == "linear_reg":
        if (option == "normal"):
            isExpert = False
            isExpertType = 'Normal'
            print("before return", isExpert, isExpertType)
            return render_template('fileupload.html', isExpert=isExpert, isExpertType=isExpertType)
        elif(option == "expert"):
            isExpert = True
            isExpertType = 'Expert'
            print("before return", isExpert, isExpertType)
            # upload_files(isExpertType)
            return render_template('fileupload.html', isExpert=isExpert, isExpertType=isExpertType)
    elif algotype == "logistic_reg":
        if (option == "normal"):
            isExpert = False
            isExpertType = 'Normal'
            print("before return", isExpert, isExpertType)
            return render_template('fileupload.html', isExpert=isExpert, isExpertType=isExpertType)
        elif(option == "expert"):
            isExpert = True
            isExpertType = 'Expert'
            print("before return", isExpert, isExpertType)
            # upload_files(isExpertType)
            return render_template('fileupload.html', isExpert=isExpert, isExpertType=isExpertType)
    elif algotype == "other":
        pass
    # return render_template('fileupload.html')





# File upload Page -----------------------------------------------------------------------------------
@app.route('/fileupload', methods=['POST'])
def upload_files():
    global file_ext
    global mydata
    global imgpath_html
    global colval
    global lr_model_final
    global lr_model_multi_final
    global lg_model_final
    global lg_expert_final

    print("We are in File upload page ............!")
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    print("My File Name : ", filename)
    if filename == '':
            flash('No selected file')
            return redirect(request.url)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]

        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            flash('File Extension not supported, Please upload TXT, CSV files only')
            return render_template('fileupload.html')
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        flash('Your file accepted by this ML tool box')
        if file_ext == ".txt":
            print(os.path.join(app.config['UPLOAD_PATH'])+"\\"+filename)
            mypath = os.path.join(app.config['UPLOAD_PATH'])+"\\"+filename
            # mydata = np.loadtxt(mypath, delimiter=',', dtype=float)
            mydata = pd.read_csv(mypath, sep = ',', header=None)
        elif file_ext == ".csv":
            mydata = pd.read_csv(os.path.join(app.config['UPLOAD_PATH'], filename))
        else:
            return "File not supported by this ML Toolbox"
        # if has_index:
        #     mydata = mydata.iloc[: , 1:]
        # if isExpert:
        # print("my_var : ", my_var)
        print("isExpertType===>: ", type(isExpertType), type(algotype))
        print(sample())
        print(mydata.head())
        colval = mydata.shape[1]
        print("colval==>", colval)
        if isExpertType == "Normal" and algotype == "linear_reg" :
            if colval == 2:
                # uni variate lr normal
                output, imgpath_html, Mean_Absolute_Error, Mean_Squared_Error, Root_Mean_Squared_Error, score, lr_model_final = lr_model_alog(mydata, file_ext)
                print("lr_model_alog ==> ", output, imgpath_html)
                return render_template('lr_normal.html', output = output, imgpath_html=imgpath_html,  Mean_Absolute_Error = Mean_Absolute_Error,  Mean_Squared_Error = Mean_Squared_Error, Root_Mean_Squared_Error = Root_Mean_Squared_Error, score=score)
            # multi variate lr normal 
            output, imgpath_html, Mean_Absolute_Error, Mean_Squared_Error, Root_Mean_Squared_Error, score, lr_model_multi_final = lr_normal_multi(mydata, file_ext)
            print("lr_model_alog ==> ", output, imgpath_html)
            return render_template('lr_normal_multi.html', output = output, imgpath_html=imgpath_html, Mean_Absolute_Error = Mean_Absolute_Error,  Mean_Squared_Error = Mean_Squared_Error, Root_Mean_Squared_Error = Root_Mean_Squared_Error, score=score)
            
        elif isExpertType == "Expert" and algotype == "linear_reg":
            if colval == 2:
                # uni variate lr expert
                return render_template('lr_expert_user_input.html')
            # multi variate lr expert
            return render_template('lr_expert_user_input.html')
        
        # -----------------------------logistic regression code-----------------------------------------------------
        # logistic regression code
        elif isExpertType == "Normal" and algotype == "logistic_reg":
            print ("col val ========>" , colval)
            if colval == 2:
                # uni variate logistic regression - Normal
                output, imgpath_html, lg_model_final = lg_model_normal(mydata, file_ext)
                print("lg_model_univariable ==> ", output, imgpath_html)
                return render_template('lg_normal.html', output = output, imgpath_html=imgpath_html)
            # multi variate logistic regression - Normal
            output, imgpath_html, Mean_Absolute_Error, Mean_Squared_Error, Root_Mean_Squared_Error, score, lr_model_multi_final = lr_normal_multi(mydata, file_ext)
            print("lg_model_alog ==> ", output, imgpath_html)
            return render_template('lg_normal_multi.html', output = output, imgpath_html=imgpath_html, Mean_Absolute_Error = Mean_Absolute_Error,  Mean_Squared_Error = Mean_Squared_Error, Root_Mean_Squared_Error = Root_Mean_Squared_Error, score=score)

        elif isExpertType == "Expert" and algotype == "logistic_reg":
            print ("col val ========>" , colval)
            if colval == 2:
                # uni variate logistic regression - Expert
                output, imgpath_html, lg_expert_final = lg_model_expert(mydata, file_ext)
                print("lg_model_univariable ==> ", output, imgpath_html)
                return render_template('lg_multi_expert.html', output = output, imgpath_html=imgpath_html)
            # multi variate logistic regression - Expert
            output, imgpath_html, lg_expert_final = lg_model_expert(mydata, file_ext)
            print("lg_model_alog ==> ", output, imgpath_html)
            return render_template('lg_multi_expert.html', output = output, imgpath_html=imgpath_html)

# Lr_normal Page -----------------------------------------------------------------------------------
@app.route('/lr_normal', methods=['POST'])
def lr_normal():
    myval = "We are in Normal mode page"
    return render_template('lr_normal.html', myval=myval)

# lr_expert_user_input Page -----------------------------------------------------------------------------------
@app.route('/lr_expert_user_input', methods=['POST'])
def expert():
    global lr_model_expert_final
    global lr_expert_multi_final
    myval = "We are in Expert mode page"
    Fit_Intercept_Val = False if request.form.getlist('Fit_Intercept')[0] == 'False' else True
    copy_X_Val = False if request.form.getlist('copy_X')[0] == 'False' else True
    n_jobs_Val = request.form.getlist('n_jobs')[0]
    if n_jobs_Val == "":
        n_jobs_Val = None
    else:
        print("n_jobs_Val :", n_jobs_Val)
        n_jobs_Val = int(n_jobs_Val)
    positive_Val = False if request.form.getlist('positive')[0] == 'False' else True

    if colval == 2:
        # uni variate lr expert
        output, imgpath_html, Mean_Absolute_Error, Mean_Squared_Error, Root_Mean_Squared_Error, score, lr_model_expert_final = lr_model_expert(mydata, file_ext, Fit_Intercept_Val, copy_X_Val, n_jobs_Val, positive_Val)
        print("output===>" , output, imgpath_html)
        return render_template('lr_expert.html', myval=myval, Fit_Intercept_Val=Fit_Intercept_Val, copy_X_Val=copy_X_Val, n_jobs_Val=n_jobs_Val, positive_Val=positive_Val, output=output, imgpath_html=imgpath_html, Mean_Absolute_Error = Mean_Absolute_Error,  Mean_Squared_Error = Mean_Squared_Error, Root_Mean_Squared_Error = Root_Mean_Squared_Error, score=score )
    # multi variate lr expert
    output, imgpath_html, Mean_Absolute_Error, Mean_Squared_Error, Root_Mean_Squared_Error, score, lr_expert_multi_final = lr_expert_multi(mydata, file_ext, Fit_Intercept_Val, copy_X_Val, n_jobs_Val, positive_Val)
    print("output===>" , output, imgpath_html)
    return render_template('lr_expert_multi.html', myval=myval, Fit_Intercept_Val=Fit_Intercept_Val, copy_X_Val=copy_X_Val, n_jobs_Val=n_jobs_Val, positive_Val=positive_Val, output=output, imgpath_html=imgpath_html, Mean_Absolute_Error = Mean_Absolute_Error,  Mean_Squared_Error = Mean_Squared_Error, Root_Mean_Squared_Error = Root_Mean_Squared_Error, score=score )


# lr_final_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_input')
def lr_normal1():
    # myval = "We are in Normal mode page"
    return render_template('lr_final_input.html')
# lr_final_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_input', methods=['POST'])
def lr_final_input():
    lr_final_input_val = float(request.form.getlist('finalinputval')[0])
    Y_pred_val = lr_model_final.predict(np.array([lr_final_input_val]).reshape(1, 1))[0]
    print("__________Done ____________", lr_final_input_val)
    return render_template('lr_final_input.html', Y_pred_val=Y_pred_val)



# lr_final_expert_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_expert_input')
def lr_expert1():
    return render_template('lr_final_expert_input.html')
# lr_final_expert_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_expert_input', methods=['POST'])
def lr_final_expert_input():
    lr_final_expert_input_val = float(request.form.getlist('finalinputval')[0])
    Y_pred_val = lr_model_expert_final.predict(np.array([lr_final_expert_input_val]).reshape(1, 1))[0]
    print("__________Done ____________", lr_final_expert_input_val)
    return render_template('lr_final_expert_input.html', Y_pred_val=Y_pred_val)



# lr_final_multi_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_multi_input')
def lr_expert2():
    # myval = "We are in Normal mode page"
    return render_template('lr_final_multi_input.html')
# lr_final_multi_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_multi_input', methods=['POST'])
def lr_final_multi_input():
    lr_final_multi_input_val1 = float(request.form.getlist('finalinputval1')[0])
    lr_final_multi_input_val2 = float(request.form.getlist('finalinputval2')[0])
    Y_pred_val = lr_model_multi_final.predict(np.array([lr_final_multi_input_val1, lr_final_multi_input_val2]).reshape(1, 2))[0]
    print("__________Done ____________", lr_final_multi_input_val1, lr_final_multi_input_val2)
    return render_template('lr_final_multi_input.html', Y_pred_val=Y_pred_val)


# lr_final_expert_multi_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_expert_multi_input')
def lr_normal2():
    # myval = "We are in Normal mode page"
    return render_template('lr_final_expert_multi_input.html')
# lr_final_multi_input Page -----------------------------------------------------------------------------------
@app.route('/lr_final_expert_multi_input', methods=['POST'])
def lr_final_expert_multi_input():
    lr_final_expert_multi_input_val1 = float(request.form.getlist('lr_final_expert_multi_input1')[0])
    lr_final_expert_multi_input_val2 = float(request.form.getlist('lr_final_expert_multi_input2')[0])
    Y_pred_val = lr_expert_multi_final.predict(np.array([lr_final_expert_multi_input_val1, lr_final_expert_multi_input_val2]).reshape(1, 2))[0]
    print("__________Done ____________", lr_final_expert_multi_input_val1, lr_final_expert_multi_input_val2)
    return render_template('lr_final_expert_multi_input.html', Y_pred_val=Y_pred_val)


# -----------------------Logistic Regression Normal Uni variable------------------------------------------

# Lg_normal Page -----------------------------------------------------------------------------------
@app.route('/lg_normal', methods=['POST'])
def lg_normal():
    myval = "We are in Logistic Regression Normal mode page"
    return render_template('lg_normal.html')

# lg_final_input Page -----------------------------------------------------------------------------------
@app.route('/lg_final_input')
def lg_normal1():
    # myval = "We are in Normal mode page"
    return render_template('lg_final_input.html')
# lg_final_input Page -----------------------------------------------------------------------------------
@app.route('/lg_final_input', methods=['POST'])
def lg_final_input():
    lg_final_input = float(request.form.getlist('finalinputval')[0])
    Y_pred_val = lg_model_final.predict(np.array([lg_final_input]).reshape(1, 1))[0]
    print("__________Done ____________", lg_final_input)
    return render_template('lg_final_input.html', Y_pred_val=Y_pred_val)


# -----------------------Logistic Regression Expert Multi variable------------------------------------------

# Lg_normal Page -----------------------------------------------------------------------------------
@app.route('/lg_multi_expert', methods=['POST'])
def lg_expert():
    myval = "We are in Logistic Regression Normal mode page"
    return render_template('lg_multi_expert.html')

# lg_final_multi_input Page -----------------------------------------------------------------------------------
@app.route('/lg_final_multi_input')
def lg_expert1():
    # myval = "We are in Normal mode page"
    return render_template('lg_final_multi_input.html')
# lg_final_multi_input Page -----------------------------------------------------------------------------------
@app.route('/lg_final_multi_input', methods=['POST'])
def lg_final_multi_input():
    lg_final_multi_input1 = float(request.form.getlist('finalinputval1')[0])
    lg_final_multi_input2 = float(request.form.getlist('finalinputval2')[0])
    Y_pred_val = lg_expert_final.predict(np.array([lg_final_multi_input1, lg_final_multi_input2]).reshape(1, 2))[0]
    print("__________Done ____________", lg_final_multi_input)
    return render_template('lg_final_multi_input.html', Y_pred_val=Y_pred_val)



# App Run -----------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
## For deploying the app use `app.run(debug=False, host="0.0.0.0", port=80)`