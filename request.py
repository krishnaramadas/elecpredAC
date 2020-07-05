# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:17:33 2020

@author: Krishna
"""
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Total Time Spent on Website':674, 'Total Visits':5, 'Page Views Per Visit':2})

print(r.json())

