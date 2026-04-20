from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import json
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# -------------------------
# LOAD MODEL + RESULTS
# -------------------------
model = joblib.load("model.pkl")

with open("results.json") as f:
    results = json.load(f)

# -------------------------
# FRONTEND UI
# -------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>PCOS AI Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body {
    background: #0f172a;
    color: #e0f7fa;
    font-family: 'Segoe UI', sans-serif;
    text-align: center;
}

h1 {
    color: #00f7ff;
    text-shadow: 0 0 15px #00f7ff;
}

.container {
    width: 75%;
    margin: auto;
}

.card {
    background: #020617;
    border: 1px solid #00f7ff;
    box-shadow: 0 0 20px #00f7ff33;
    padding: 20px;
    margin-top: 25px;
    border-radius: 15px;
}

input, select {
    padding: 10px;
    margin: 10px;
    border-radius: 5px;
    border: none;
    width: 200px;
}

button {
    padding: 10px 20px;
    background: #00f7ff;
    color: black;
    border: none;
    cursor: pointer;
    border-radius: 5px;
    font-weight: bold;
}

button:hover {
    background: #00c4cc;
}

.result {
    font-size: 22px;
    margin-top: 10px;
}

.bmi {
    margin-top: 10px;
    font-size: 18px;
}

canvas {
    margin-top: 20px;
}
</style>
</head>

<body>

<h1>🚀 PCOS AI Dashboard</h1>

<div class="container">

<div class="card">
<h2>📊 Model Metrics</h2>
<p id="metrics"></p>

<select id="graphType" onchange="drawChart()">
<option value="bar">Bar</option>
<option value="line">Line</option>
<option value="radar">Radar</option>
</select>

<canvas id="chart"></canvas>
</div>

<div class="card">
<h2>📉 Confusion Matrix</h2>
<canvas id="heatmap"></canvas>
</div>

<div class="card">
<h2>🧬 Predict PCOS</h2>

<input id="age" placeholder="Age">
<input id="weight" placeholder="Weight (kg)">
<input id="height" placeholder="Height (cm)">

<br>

<button onclick="calculateBMI()">Calculate BMI</button>

<p class="bmi" id="bmi"></p>

<button onclick="predict()">Predict</button>

<p class="result" id="result"></p>
<p class="result" id="risk"></p>

</div>

</div>

<script>

let chart;
let metricsData;

// Load metrics
fetch("/metrics")
.then(res => res.json())
.then(data => {
    metricsData = data;

    document.getElementById("metrics").innerHTML =
        "Accuracy: " + data.accuracy.toFixed(2) + "<br>" +
        "Precision: " + data.precision.toFixed(2) + "<br>" +
        "Recall: " + data.recall.toFixed(2) + "<br>" +
        "F1 Score: " + data.f1_score.toFixed(2);

    drawChart();
    drawHeatmap(data.confusion_matrix);
});

// Graph
function drawChart() {
    let type = document.getElementById("graphType").value;
    if (chart) chart.destroy();

    chart = new Chart(document.getElementById("chart"), {
        type: type,
        data: {
            labels: ["Accuracy","Precision","Recall","F1"],
            datasets: [{
                data: [
                    metricsData.accuracy,
                    metricsData.precision,
                    metricsData.recall,
                    metricsData.f1_score
                ],
                borderColor: "#00f7ff",
                backgroundColor: "rgba(0,247,255,0.3)"
            }]
        }
    });
}

// SIMPLE HEATMAP (no plugin needed)
function drawHeatmap(cm) {
    let ctx = document.getElementById("heatmap").getContext("2d");

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["TN","FP","FN","TP"],
            datasets: [{
                label: "Confusion Matrix",
                data: [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
                backgroundColor: [
                    "#00ffcc","#ff0066","#ffcc00","#00ff00"
                ]
            }]
        }
    });
}

// BMI
let bmiValue = 0;

function calculateBMI() {
    let weight = parseFloat(document.getElementById("weight").value);
    let height = parseFloat(document.getElementById("height").value)/100;

    bmiValue = weight/(height*height);

    let cat = bmiValue<18.5 ? "Underweight" :
              bmiValue<25 ? "Normal" : "Overweight";

    document.getElementById("bmi").innerText =
        "BMI: " + bmiValue.toFixed(2) + " (" + cat + ")";
}

// Predict
function predict() {
    let age = parseFloat(document.getElementById("age").value);
    let weight = parseFloat(document.getElementById("weight").value);

    let features = [age, bmiValue, weight];

    fetch("/predict", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({features:features})
    })
    .then(res=>res.json())
    .then(data=>{
        document.getElementById("result").innerText =
            data.prediction==1 ? "⚠️ PCOS Detected" : "✅ No PCOS";

        document.getElementById("risk").innerText =
            "Risk Score: " + data.risk + "%";
    });
}

</script>

</body>
</html>
"""

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/metrics")
def metrics():
    return jsonify(results)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1] * 100

    return jsonify({
        "prediction": int(prediction),
        "risk": round(prob,2)
    })

# -------------------------
# RUN (DEPLOYMENT READY)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)