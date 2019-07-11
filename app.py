from flask import Flask, render_template, request
import naivebayes2 as nb2
import numpy as np

app = Flask(__name__)


@app.route("/")
def Heart():
    return render_template("Heart.html", result=result)


@app.route("/Heart", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        result = request.form
        input = [
            [
                request.form["age"],
                request.form["sex"],
                request.form["chest"],
                request.form["pressure"],
                request.form["cholestoral"],
                request.form["sugar"],
                request.form["electro"],
                request.form["maximum"],
                request.form["exercise"],
                request.form["oldpeak"],
                request.form["slope"],
                request.form["number"],
                request.form["thal"],
            ]
        ]
        input = np.array(input, float)
        predictions = nb2.nb.predict(input)[0]
        print("predictions:", predictions)

        return render_template("Heart.html", result=result, predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

