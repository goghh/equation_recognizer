from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import JsonResponse

from er.Latex.Latex import Latex
from os import listdir
from skimage import io

import tensorflow as tf
import numpy as np
import uuid
import os
import cv2


def index(request):
    return render(request, 'index.html')


def recognition(request):
    tf.reset_default_graph()
    mean_train = np.load("er/train_images_mean.npy")
    std_train = np.load("er/train_images_std.npy")

    model = Latex("er/model", mean_train, std_train, plotting=False)
    uploaded_file = handle_uploaded_file(request.FILES.get('file'))
    path = "media/media/" + uploaded_file['filename']
    image = io.imread(path)
    print(image.dtype)
    image = cv2.convertScaleAbs(image)
    latex = model.predict(image)

    result = {"file": latex['equation'], "img": path}

    return render(request, 'index.html', {'result': result})


def handle_uploaded_file(file):
    ext = file.name.split('.')[-1]
    filename = "%s.%s" % (uuid.uuid4(), ext)
    save_path = os.path.join(settings.MEDIA_ROOT, 'media', filename)
    path = default_storage.save(save_path, file)

    return {"filename": filename, 'path': path}
