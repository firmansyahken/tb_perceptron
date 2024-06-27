from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

models = pickle.load(open("model.pkl", "rb"))
model = models['model']
min_umur = models['min_umur'] 
max_umur = models['max_umur'] 

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/requirements")
def Requirements():
    return render_template("requirements.html")

@app.route("/preprocessing")
def Preprocessing():
    return render_template("preprocessing.html")

@app.route("/modeling")
def Modeling():
    return render_template("modeling.html")

@app.route("/test")
def Test():
    return render_template("test.html")

@app.route("/classification", methods=["POST"])
def Classification():
    req = request.form.values()
    new_data = []
    for index, x in enumerate(req):
        if index == 0:
            age = float(x)
            age_normalized = (age - min_umur) / (max_umur - min_umur)
            new_data.append(age_normalized)
        else:
            new_data.append(float(x))
    x = [np.array(new_data)]
    y = model.predict(x)
    result = ""
    if y[0] == 1:
         result = "Paru"
    else:
        result = "Ekstra Paru"
    return render_template("result.html", result_text = result, req_text=request.form.values())
if __name__ == "__main__":
    app.run(debug=True)