from django.db import models
from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()

connection_String = os.getenv("CONSTRING")

client = MongoClient(connection_String)
db = client['history']


class UploadedFile(models.Model):
    file = models.FileField(upload_to='pdfs/')
