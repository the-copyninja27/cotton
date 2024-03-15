"""
URL configuration for cotton project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render , HttpResponse
from PIL import Image
import numpy as np
from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.load_weights('models/cotton/model.h5')

@csrf_exempt
def index(request):
    print(request.POST)
    crop_image = request.FILES.get('crop_image')
    print(request.FILES)
    index_to_class = {
        0: 'Aphids',
        1: 'Army worml̥',
        2: 'Bacterial Blight',
        3: 'Healthy',
        4: 'Powdery Mildew',
        5: 'Target spot'
    }
    pillowobj = Image.open(crop_image)
    pillowobj = pillowobj.resize((150, 150))
    print(pillowobj)
    img_array = img_to_array(pillowobj)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    result = index_to_class[predicted_class[0]]
    return HttpResponse(result)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detect',index)
]
