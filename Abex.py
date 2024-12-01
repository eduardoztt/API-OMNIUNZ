from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import base64

app = Flask(__name__)

classes = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
    "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"
]

def create_model():
    IMAGE_SHAPE = (192, 192, 3) 
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(classes), activation='softmax')  # Número de classes
    ])
    
    return model

model = create_model()

#caminho_pesos = './pesos_mobilenetv2.weights.h5'  

try:
    model.load_weights('./pesos_mobilenetv2.weights.h5', skip_mismatch=True)
    print("Pesos carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar os pesos adicionais: {str(e)}")

def preprocess(img):
    img = img.resize((192, 192))  
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  
    return img

@app.route("/predict", methods=['POST'])
def predict():
    dados = request.json

    if dados is None or 'imagem' not in dados:
        return jsonify({"erro": "Nenhuma imagem foi enviada"}), 400
    
    try:
        img_data = dados['imagem']
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]
        
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data))

        img = img.resize((192, 192))
    except Exception as e:
        return jsonify({"erro": f"Erro ao processar a imagem. Erro -> {str(e)}"}), 400
    
    try:
        img = preprocess(img)
        prediction = model.predict(img)
        top_indices = prediction.argsort()[0][-1:][::-1]  
        top_classes = [{"classe": classes[i], 
                        "probabilidade": f"{prediction[0][i] * 100:.2f}%"} 
                       for i in top_indices]
    
    except Exception as e:
        return jsonify({"erro": f"Erro ao inferir o resultado: {str(e)}"}), 500

    return jsonify(top_classes)

if __name__ == '__main__':
    app.run(debug=True)
