# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:59:42 2022

@author: San
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_proba():
    url_path = '/predict/?contract=Month-to-month&dependents=Yes&deviceprotection=Yes&gender=Male&internetservice=Fiber%20optic&multiplelines=Yes&onlinebackup=Yes&onlinesecurity=Yes&paperlessbilling=Yes&partner=Yes&paymentmethod=Electronic%20check&phoneservice=Yes&seniorcitizen=Yes&streamingmovies=Yes&streamingtv=Yes&techsupport=Yes&tenure=12&monthlycharges=11&totalcharges=144'
    response = client.get(url_path)
    assert response.status_code == 200
    assert response.json() == {"default probability": 0.6479119658302044}
    
def test_predict_proba2():
    url_path = '/predict/?contract=Month-to-month&dependents=No&deviceprotection=No%20internet%20service&gender=Male&internetservice=No&multiplelines=No%20phone%20service&onlinebackup=No%20internet%20service&onlinesecurity=No%20internet%20service&paperlessbilling=No&partner=No&paymentmethod=Electronic%20check&phoneservice=No&seniorcitizen=No&streamingmovies=No%20internet%20service&streamingtv=No%20internet%20service&techsupport=No%20internet%20service&tenure=1&monthlycharges=19&totalcharges=999'
    response = client.get(url_path)
    assert response.status_code == 200
    assert response.json() == {"default probability": 0.47607642962407437}