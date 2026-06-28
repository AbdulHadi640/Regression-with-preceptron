# Custom Perceptron from Scratch: Linear & Multiple Regression

A foundational machine learning project implementing a single-layer Perceptron neural network from scratch using Python and NumPy. This project demonstrates vectorization, gradient descent optimization, feature scaling, and 3D visualization without relying on high-level ML frameworks like Scikit-Learn or TensorFlow.

## 🚀 Key Features
- **Mathematical Vectorization:** Implemented custom forward propagation ($W \cdot X + b$) and backpropagation routines utilizing NumPy matrix multiplication and broadcasting.
- **Feature Normalization:** Built column-wise Z-score standardization (`axis=0`) to handle disparate feature scales and eliminate gradient explosion (`inf`/`nan` errors).
- **Dual Architecture:** - *Simple Linear Regression:* Modeling TV marketing spend vs. Sales.
  - *Multiple Linear Regression:* Modeling house prices using multiple features (`GrLivArea`, `OverallQual`).
- **3D Spatial Visualization:** Rendered a 3D regression plane cutting through multi-variable training clusters using `matplotlib`.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib

## 📈 Learning Highlights
- Successfully diagnosed and corrected a gradient explosion by adjusting hyperparameter step-sizes (tuning learning rate down to `0.001`).
- Mastered matrix alignment dimensions, ensuring shape transitions match mathematical criteria across multi-feature datasets.
