# Particle-Detector-Hit-Simulation-and-Track-Visualization
# Particle Detector Hit Simulation and Track Visualization

## Project Description
This project simulates how particles leave hits in a detector during high-energy physics experiments.

When particles move through detectors, they leave signals in different detector layers. By analyzing these hits, physicists reconstruct particle tracks.

This project simulates detector hits and visualizes particle tracks using Python.

## Objectives
- Generate simulated detector hits
- Visualize particle tracks
- Understand detector measurements
- Perform simple track analysis

## Tools Used
- Python
- NumPy
- Matplotlib

## Dataset Description
Each event includes:
- detector layer number
- x and y hit positions
- particle id

## Possible Extensions
- Track reconstruction algorithms
- Noise simulation
- Machine learning track detection
# Dataset (events.csv)
energy,momentum,angle,event_type
110,70,0.50,1
95,55,0.40,0
210,150,1.20,1
60,35,0.25,0
180,120,0.90,1
75,45,0.38,0
205,160,1.15,1
68,40,0.32,0
155,98,0.75,1
50,25,0.20,0
130,85,0.60,1
72,42,0.35,0
# Analysis Code (analysis.py)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("events.csv")

# Features and labels
X = data[["energy", "momentum", "angle"]]
y = data["event_type"]

# Train classifier
model = LogisticRegression()
model.fit(X, y)

# Accuracy
accuracy = model.score(X, y)
print("Model accuracy:", accuracy)

# Predictions
data["prediction"] = model.predict(X)

# Energy distribution
plt.figure()
plt.hist(data["energy"], bins=6)
plt.xlabel("Energy (GeV)")
plt.ylabel("Counts")
plt.title("Energy Distribution of Events")
plt.savefig("energy_histogram.png")
plt.show()

# Energy vs momentum plot
plt.figure()
plt.scatter(data["energy"], data["momentum"])
plt.xlabel("Energy (GeV)")
plt.ylabel("Momentum")
plt.title("Collision Events")
plt.savefig("energy_momentum_scatter.png")
plt.show()
