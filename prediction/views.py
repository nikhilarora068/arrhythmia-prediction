from django.shortcuts import render, render
import base64



import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

model = tensorflow.keras.models.load_model('/home/Linux/Predict_types_of_arrhythmia.h5')
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

        # 'N','A','V','P','R','f','L'
        N = pred[0][0]
        A = pred[0][1]
        V = pred[0][2]
        P = pred[0][3]
        R = pred[0][4]
        L = pred[0][6]

        probability = -1

        type = ""
        if (N < 0.5):
            probability = max(A, L, P, R, V)
            if (probability == A):
                type = 'A'
            elif (probability == L):
                type = 'L'
            elif (probability == P):
                type = 'P'
            elif (probability == R):
                type = 'R'
            else:
                type = 'V'
        else:
            probability = N

        # probability = pred[0][0]

        print("probability = ", probability)

        prediction = ""
        status = -1
        accuracy = 0

        if probability > 0.5:
            prediction = "NO ARRHYTHMIA"
            status = 1
            accuracy = ("%.2f" % (probability * 100))
            print("%.2f" % (probability * 100) + "% normal")
        else:
            prediction = "ARRHYTHMIA"
            status = 0
            accuracy = ("%.2f" % ((1 - probability) * 100))
            print("%.2f" % ((1 - probability) * 100) + "% arrhythmia")




        print("DONEEEE")
        if (status == 1):
            context = {
                'image': input_image,
                'result': prediction,
                'status': status,
                'accuracy': accuracy
            }
        else:
            context = {
                'image': input_image,
                'result': prediction,
                'status': status,
                'types': type,
                'accuracy': accuracy
            }
        # context = {
        #     'image': input_image,
        #     'result': prediction,
        #     'status': status,
        #     'accuracy': accuracy
        # }


        return render(request, 'index.html', context)

    return render(request, 'index.html')




def temp(request):
    return render(request, 'test.html')


