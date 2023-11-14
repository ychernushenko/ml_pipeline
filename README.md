# Activate virtual environment and install dependencies Linux/MacOS
```
python -m venv venv  
source venv/bin/activate
pip install -r requirements.txt  
```

# Download dataset Linux/MacOS
- curl https://surfdrive.surf.nl/files/index.php/s/LDwpIdG7HHkQiOs/download --output dataset.zip  
- unzip dataset.zip  

# Run  
```
python ./src/ml_pipeline.py --data_root_dir './dataset'
```
expected output:  
```
Accuracy: 0.98

Classification Report:
              precision    recall  f1-score   support

     Jewelry       0.99      0.98      0.98       161
     Kitchen       0.98      0.99      0.98       163

    accuracy                           0.98       324
   macro avg       0.98      0.98      0.98       324
weighted avg       0.98      0.98      0.98       324
```