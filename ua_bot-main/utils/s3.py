import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def download_dir(client, resource, dist, bucket, local='/tmp'):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(local, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if not file.get('Key').endswith('/'):
                resource.meta.client.download_file(
                    bucket, file.get('Key'), dest_pathname
                )


def download_s3_directory(bucket_name, s3_dir, local_dir):
    config = boto3.session.Config(signature_version='s3v4')
    client = boto3.client(
        's3',
        config=config,
        region_name=os.getenv('AWS_REGION'),
    )
    resource = boto3.resource('s3', region_name=os.getenv('AWS_REGION'))
    download_dir(client, resource, s3_dir, bucket_name, local_dir)
