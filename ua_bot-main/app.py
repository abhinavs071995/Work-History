import os
from pathlib import Path

from flask import Flask, request
from flask_crontab import Crontab

from chatbot_v3_get_response import get_response
from creating_pickle import create_vectors
from utils import download_s3_directory

app = Flask(__name__)
crontab = Crontab(app)


@app.route('/ping', methods=['GET'])
def ping():
    return {'message': 'pong'}


@app.route('/', methods=['POST'])
def chatbot():
    data = request.get_json()

    user = data.get('user', '')
    message_history = data.get('message_history', [])
    status = data.get('status', '')

    ai_response, message_history, status = get_response(user, message_history, status)

    return {
        'ai_response': ai_response,
        'message_history': message_history,
        'status': status,
    }


# Run cron job to fetch vectors from S3 every 24 hours
# @crontab.job(minute=0, hour=0)
# def fetch_vectors():
#     bucket_name = os.getenv('S3_BUCKET')
#     s3_dir = os.getenv('S3_DIRECTORY')
#     local_dir = Path.cwd() / 'vectors'
#     download_s3_directory(bucket_name, s3_dir, local_dir)


@crontab.job(minute=0, hour=0)
def fetch_vectors():
    create_vectors()
