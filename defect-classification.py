import keras
import io
import base64
from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.debug = True
def get_model():
	global model
	model = load_model('new-model.h5')
	print(' * Model is loaded!')

def preprocess_image(image, target_size):
	image = image.convert('L')
	image = image.resize(target_size)
	image = img_to_array(image)/255.
	image = np.expand_dims(image, axis = 0)

	return image
def predict_class(predict):
	L = ['Dark Spot','Fold strip','Gel','Lump','Streak','Wrinkle']
	pred = np.amax(predict)
	if pred > 0.75:
		index_pred = np.argmax(predict)
		text = L[index_pred]
	else:
		text = 'Unknow'
	return text

print(' * Loading Keras model...')
get_model()
graph = tf.get_default_graph()
@app.route('/predict', methods = ['POST'])
def predict():
	global graph
	with graph.as_default():

		message = request.get_json(force = True)
		encoded = message['image']
		decoded = base64.b64decode(encoded)
		image = Image.open(io.BytesIO(decoded))
		processed_image = preprocess_image(image, target_size=(192,192))

		prediction = model.predict(processed_image).tolist()
		result = predict_class(prediction)
		response = {
			'prediction' : {
				'DarkSpot' : prediction[0][0]*100.,
				'Foldstrip' : prediction[0][1]*100.,
				'Gels': prediction[0][2]*100.,
				'Lumps': prediction[0][3]*100.,
				'Streak': prediction[0][4]*100.,
				'Wrinkle': prediction[0][5]*100.,
				'result' : result
			}
		}
		return jsonify(response)
app.run()
