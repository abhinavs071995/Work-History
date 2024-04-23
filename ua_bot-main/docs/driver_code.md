```python
import requests
import timeit

global user,message_history,status
user = input("You: ")

message_history=[]
status=""
while user!="exit":
    url = 'http://127.0.0.1:5000/'
    url = 'https://chatbot.uniacco.com/'
    #url='http://13.201.161.64:8889/'
    #url='http://metabase.unischolars.xom:8889/'
    #url=  'http://172.31.32.137:5000/'
    payload = {
        'user': user,
        'message_history': message_history,
        'status': status
    }
    # print(payload)
    t_0 = timeit.default_timer()
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        ai_response = data['ai_response']
        message_history = data['message_history']
        status = data['status']
        t_1 = timeit.default_timer()
        print(f'sent {t_1-t_0} seconds ago')
        print(f'AI Response: {ai_response}')
        user = input("You: ")
#         print(f'Updated Message History: {message_history}')
#         print(f'Status: {status}')
    else:
        print(f'Request failed with status code: {response.status_code}')
        print(response.text)

```

# To download a directory from S3
```python
import os
from pathlib import Path
from utils import download_s3_directory
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv('S3_BUCKET')
# s3_dir = os.getenv('S3_DIRECTORY')
s3_dir = 'vectors/cities'
local_dir = Path.cwd() / 'vectors' / 'cities'
(bucket_name, s3_dir, local_dir)

download_s3_directory(bucket_name, s3_dir, local_dir)
```
```python
import os
from pathlib import Path
from utils import download_s3_directory
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv('S3_BUCKET')
# s3_dir = os.getenv('S3_DIRECTORY')
s3_dir = 'vectors/cities'
local_dir = Path.cwd() / 'vectors' / 'cities'
(bucket_name, s3_dir, local_dir)

download_s3_directory(bucket_name, s3_dir, local_dir)
```
