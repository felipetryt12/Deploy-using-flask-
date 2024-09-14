from flask import Flask, request, render_template
import joblib
import os
import numpy as np

app = Flask(__name__)

modelo_path = os.path.join('models', 'modelo_regresion_lineal.pkl')
modelo = joblib.load(modelo_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediccion = modelo.predict(final_features)

    return render_template('index.html', prediction_text=f'La predicción de la calificación es: {prediccion[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
