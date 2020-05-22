from django.shortcuts import render, render
import base64



import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# model = tensorflow.keras.models.load_model('/home/Linux/Predict_types_of_arrhythmia.h5')
model = tensorflow.keras.models.load_model('/Users/NikhilArora/desktop/Major-II/breakthrough/model3.h5')
print('loaded model')


# Create your views here.

def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        inImg = request.FILES['myfile'].read()

        encoded = base64.b64encode(inImg).decode('ascii')
        mime = "image/jpg"
        mime = mime + ";" if mime else ";"
        input_image = "data:%sbase64,%s" % (mime, encoded)

        image = load_img(request.FILES['myfile'], target_size=(128, 128))

        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.

        pred = model.predict(image.reshape((1, 128, 128, 3)))
        pred_class = pred.argmax(axis=-1)

        # folder order 'L' 'N' 'P' 'R' 'V'

        L = pred[0][1]
        N = pred[0][2]
        P = pred[0][3]
        R = pred[0][4]
        V = pred[0][6]

        type = ""
        status = 0
        probability = max(L, N, P, R, V)

        if probability == N:
            type = 'Normal heartbeat'
            status = 1
        elif probability == L:
            type = 'Left bundle branch block beat'
        elif probability == P:
            type = 'Paced beat'
        elif probability == R:
            type = 'Right bundle branch block beat'
        elif probability == V:
            type = 'Premature ventricular contraction'

        print("probability = ", probability)

        context = {
            'image': input_image,
            'status': status,
            'accuracy': probability,
            'type': type
        }

        return render(request, 'index.html', context)

    return render(request, 'index.html')




def temp(request):
    return render(request, 'test.html')
