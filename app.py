from flask import Flask, request, jsonify, render_template
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset and train the model
iris = datasets.load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    prediction = model.predict([features])
    species_name = iris.target_names[prediction[0]]
    return jsonify({'species': species_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)