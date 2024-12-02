from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import base64

app = Flask(__name__)

# Definindo as classes
classes = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
    "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"
]

# Função para criar o modelo
def create_model():
    IMAGE_SHAPE = (192, 192, 3)  # Tamanho das imagens
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

# Criar e carregar o modelo
model = create_model()

# Carregar os pesos do modelo
try:
    model.load_weights('./pesos_mobilenetv2.weights.h5', skip_mismatch=True)
    print("Pesos carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar os pesos adicionais: {str(e)}")

# Função de pré-processamento da imagem
def preprocess(img):
    img = img.resize((192, 192))  # Redimensiona a imagem
    img = img_to_array(img)  # Converte para array
    img = np.expand_dims(img, axis=0)  # Expande as dimensões para o formato do modelo
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Pré-processamento específico
    return img

# Endpoint para fazer a previsão
@app.route("/predict", methods=['POST'])
def predict():
    dados = request.json

    # Verificar se a imagem foi enviada
    if dados is None or 'imagem' not in dados:
        return jsonify({"erro": "Nenhuma imagem foi enviada"}), 400
    
    try:
        img_data = dados['imagem']
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]  # Remove a parte do prefixo base64
        
        # Decodifica a imagem em base64
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data))

        # Garantir que a imagem foi carregada corretamente
        if img is None:
            return jsonify({"erro": "Não foi possível carregar a imagem"}), 400

    except Exception as e:
        return jsonify({"erro": f"Erro ao processar a imagem. Erro -> {str(e)}"}), 400
    
    try:
        # Pré-processamento da imagem
        img = preprocess(img)
        # Faz a previsão
        prediction = model.predict(img)

        # Ordenar as previsões e pegar as 3 classes mais prováveis
        top_indices = prediction.argsort()[0][-3:][::-1]  # Pegando as 3 classes mais prováveis
        top_classes = [{"classe": classes[i], 
                        "probabilidade": f"{prediction[0][i] * 100:.2f}%"} 
                       for i in top_indices]
    
    except Exception as e:
        return jsonify({"erro": f"Erro ao inferir o resultado: {str(e)}"}), 500

    # Retorna as top 3 classes
    return jsonify(top_classes)

# Rodar o servidor
if __name__ == '__main__':
    app.run(debug=True)
