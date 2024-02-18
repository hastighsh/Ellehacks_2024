from __future__ import print_function
import json

from flask import request

from flask import Flask, render_template,redirect,url_for,request


from Prediction import mainRun

app = Flask(__name__)
@app.route('/frontend/projections/projections.html', methods=['POST','GET'])
def page():
    
    if request.method=="POST":
        city_data=request.form["location"]
        date=request.form["forecast"]
       
    
        return redirect(url_for("home",location_data=city_data,date=date))
    else: 
        return render_template("/frontend/projections/projections.html")
@app.route("/<location_data>/<date>",methods=['POST','GET'])
def home(location_data,date):
     
     output = mainRun(location_data,date)
    
     return render_template("projection_output.html",location=location_data,date_data=date,prediction_return=output) 