import os
import zipfile

import boto3

from ultralytics import YOLO


bucket_name = None
sns_topic_arn = None


# download data
s3_client = boto3.client('s3')

s3_client.download_file(bucket_name, 'data.zip', 'data.zip')

with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

# train the model

model = YOLO("yolov8n.pt")

results = model.train(data="config.yaml", epochs=20)

# upload results into s3 bucket

with zipfile.ZipFile('./runs.zip', 'w') as zip:
    for path, directories, files in os.walk('./runs'):
        for file in files:
            file_name = os.path.join(path, file)
            zip.write(file_name)

s3_client.upload_file('runs.zip', bucket_name, 'runs.zip')

# send sns

sns_client = boto3.client('sns', region_name='us-east-1')

response = sns_client.publish(
    TargetArn=sns_topic_arn,
    Message="Training completed !"
)

# shutdown instance
os.system('sudo shutdown -h now')
