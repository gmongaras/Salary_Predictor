from flask import Flask, render_template, request, session
from flask import request
import complete as c
import numpy as np

app = Flask(__name__)
# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route('/', methods=["GET", "POST"])
def index():
    # If the get method was returned, go to the page.
    if request.method == "GET":
        return render_template('index.html', pred=-1)
    else:
        # Get all the data from the form
        # Remember hsc_s and degree_t are needed
        data = {"gender": [str(request.form.get("gender"))],
                "ssc_p": [request.form.get("ssc_p")],
                "ssc_b": [str(request.form.get("ssc_b"))],
                "hsc_p": [request.form.get("hsc_p")],
                "hsc_b": [str(request.form.get("hsc_b"))],
                "hsc_s": [str(request.form.get("hsc_s"))], #Required
                "degree_p": [request.form.get("degree_p")],
                "degree_t": [str(request.form.get("degree_t"))], #Required
                "workex": [str(request.form.get("workex"))],
                "etest_p": [request.form.get("etest_p")],
                "specialisation": [str(request.form.get("specialisation"))],
                "mba_p": [request.form.get("mba_p")],
                "status": ["Placed"]
               }
        # Change all none values to Nan so it can be used
        # in the prediction.
        for key in data:
            if data[key] == ['']:
                data[key] = [np.nan]

        # Get the predictions
        pred1, pred2, pred3 = c.getPred(data)

        # Convert the predictions to a dictionary
        predictions = {"pred1":pred1, "pred2":pred2, "pred3":pred3}

        # Add all predictions greater than 0
        predictionsCpy = predictions.copy()
        for k in predictionsCpy:
            if predictionsCpy[k] <= 0:
                predictions.pop(k)

        # Average the values
        if len(predictions) != 0:
            pred = 0 # Average of all predictions greater than 0
            for k in predictions:
                pred += predictions[k]
            pred /= len(predictions)
        else:
            pred = 0

        pred = round(pred, 2)

        #return render_template('index.html', scroll="pred", pred1=pred1, pred2=pred2, pred3=pred3)
        return render_template('index.html', scroll="pred", pred=pred)

if __name__ == '__main__':
    app.debug = True
    app.run()
