
# Gender Detection – Machine Learning Project

This project focuses on building a robust gender classification system using various machine learning techniques. The models are trained and evaluated on low-dimensional embeddings of face images, categorized into male and female classes.

## 📊 Dataset

The dataset contains:
- **Training Set:** 2400 samples (720 male, 1680 female)
- **Test Set:** 6000 samples (4200 male, 1800 female)
- Each sample includes **12 features** derived from image embeddings
- Highly imbalanced classes in both training and test sets

## 🔧 Tools & Technologies

- Python (NumPy, SciPy, Matplotlib)
- Gaussian Classifiers (Full, Diagonal, Tied)
- Logistic Regression (with regularization)
- Support Vector Machines (Linear, Polynomial, RBF)
- Gaussian Mixture Models (GMM)
- Principal Component Analysis (PCA)
- Score Calibration

## 🔍 Methodology

1. **Preprocessing:**
   - Applied Z-normalization to center and scale features
   - Explored PCA (tested dimensions from m=12 downwards)

2. **Model Training & Evaluation:**
   - Used both Single-Split and K-Fold Cross-Validation (K=5)
   - Evaluated models using minDCF (Minimum Detection Cost Function)
   - Balanced and unbalanced priors tested

3. **Score Calibration:**
   - Applied linear score transformation for well-calibrated outputs
   - Compared minDCF and actDCF (actual detection cost)

4. **Best Models:**
   - **GMM (Tied Covariance, 4 components)**
   - **Logistic Regression (lambda = 1e-5)**
   - **Linear SVM (C = 10, unbalanced)**
   - **MVG (Tied Full Covariance)**

## ✅ Key Findings

- **Linear models consistently outperformed quadratic models**
- PCA did not significantly improve performance beyond m=11
- GMM models provided best performance across minDCF and actDCF
- Score calibration substantially improved model reliability

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yurnero14/GenderDetection
   cd GenderDetection
   ```

2. Install required packages:
   ```bash
   pip install numpy scipy matplotlib
   ```

3. Run training and evaluation:
   ```bash
   python main.py
   ```

