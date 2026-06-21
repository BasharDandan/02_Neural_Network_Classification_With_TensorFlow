# Visualizing and Classifying Circular Data with TensorFlow

This notebook serves as a practical introduction to writing deep neural networks for classification problems using **TensorFlow**. In data science, classification tasks are used to categorize data points into distinct labels (e.g., separating object types, fraud detection, or disease diagnosis). 

## 📋 Types of Classification Problems Addressed
* **Binary classification:** Classifying something as one of two options (e.g., Yes/No, 0/1).
* **Multiclass classification:** Classifying something as one of more than two options (e.g., Cat/Dog/Bird).
* **Multilabel classification:** Classifying an item where it can belong to multiple categories simultaneously.

---

## 🛠️ Project Structure & Workflow

The notebook contains the following steps to build, analyze, and visualize a synthetic standard dataset:

### 1. Generating Data
Using `scikit-learn`, we generate a synthetic dataset of concentric circles containing 1,000 sample points with custom random seeds and noise values to simulate real-world data imperfections.

```python
from sklearn.datasets import make_circles

# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                     noise=0.03,
                     random_state=42)


 Inspecting the Generated ArraysThe raw geometric data is structured as multi-dimensional coordinate arrays.
 X contains the positional features ($X_0$ and $X_1$).
 y contains the target labels (0 for the outer circle, 1 for the inner circle).


Tabular Data Integration
To make the data human-readable, the structural arrays are formatted into a structured pandas DataFrame.


Interactive Geoplotting & Visualization
Because raw numbers are hard to parse at a glance, the data is projected visually onto a 2D scatter plot canvas to reveal the non-linear relationship between the concentric boundary rings.

🚀 Future Roadmap: Modeling with TensorFlow
The next stages of this project involve deploying deep learning models to predict boundaries:
Model Architecture: Creating standard Input, Hidden (Dense), and Output layers.
Activation Functions: Implementing non-linear activations like ReLU and Sigmoid to successfully draw a curved classification boundary.
Evaluation Metrics: Tracking Loss curves and precision via Binary Crossentropy.

🧰 Dependencies
Ensure you have the following packages installed to run this workspace:
tensorflow
pandas
scikit-learn
matplotlib


