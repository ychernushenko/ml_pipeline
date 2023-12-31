# Activate virtual environment and install dependencies Linux/MacOS
```
python -m venv venv  
source venv/bin/activate
pip install -r requirements.txt  
```

# Download dataset Linux/MacOS
- curl https://surfdrive.surf.nl/files/index.php/s/LDwpIdG7HHkQiOs/download --output dataset.zip  
- unzip dataset.zip  

# Run tests  
Only several tests are provided  
```
pytest ./tests/unit_tests.py
```

# Run  
```
python ./src/ml_pipeline.py --data_root_dir './dataset'
```
expected output:  
```
Accuracy: 0.99

Classification Report:
              precision    recall  f1-score   support

     Jewelry       0.99      0.99      0.99       162
     Kitchen       0.99      0.99      0.99       189

    accuracy                           0.99       351
   macro avg       0.99      0.99      0.99       351
weighted avg       0.99      0.99      0.99       351
```