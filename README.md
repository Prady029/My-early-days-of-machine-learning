# My Early Days of Machine Learning 🌱

*A nostalgic collection of ML implementations from 2018 — built from first principles when I was just starting out.*

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.20+-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

This repository documents my **first steps into Machine Learning** back in 2018. The code represents naive but genuine attempts to understand and implement core ML algorithms from scratch — before I knew about proper software engineering practices, vectorization, or production ML.

> ⚠️ **Disclaimer**: These are learning exercises, not production-ready code. They contain hardcoded paths, manual implementations, and beginner mistakes — preserved exactly as written for posterity.

---

## 📚 Notebooks & Files

| File | Topic | Key Concepts |
|------|-------|--------------|
| `my first linear regression.ipynb` | Simple Linear Regression | Gradient descent from scratch, hypothesis function, cost function, learning rate |
| `my multiple linear regression.ipynb` | Multiple Linear Regression | Multi-feature regression, feature scaling, normal equation |
| `my lasso.ipynb` | Lasso Regression (L1) | L1 regularization, sparsity, coordinate descent |
| `elastic.ipynb` | Elastic Net | L1 + L2 regularization combined, hyperparameter tuning |
| `polynomial.ipynb` | Polynomial Regression | Feature engineering, polynomial features, overfitting/underfitting |
| `Kmeans.ipynb` | K-Means Clustering | Unsupervised learning, centroid initialization, elbow method |
| `NaiveBayes.py` | Naive Bayes Classifier | Probabilistic classification, Bayes theorem, conditional independence |
| `Xor.py` | XOR Problem | Neural network basics, non-linear separability |
| `Bioinformatics_String_Matching_assignment.ipynb` | String Matching | Bioinformatics application, sequence alignment |
| `Gradient_descent_notes/` | Gradient Descent Theory | Mathematical derivations, convergence analysis |
| `ML_Notes/` | Study Materials | 40+ PDFs/PPTs on ML theory, algorithms, and research papers |

---

## 🔬 Learning Journey Highlights

### 1. **Linear Regression from Scratch**
```python
# Manual gradient descent implementation
def hypo(theta, x):
    return theta[0] + theta[1] * x

def cost(theta, x, y):
    m = len(x)
    return (1/(2*m)) * np.sum((hypo(theta, x) - y)**2)

def linreg(x, y, alpha, iternum):
    th = [0, 0]
    m = len(x)
    cos = []
    for i in range(iternum):
        temp[0] = th[0] - alpha * (sum(th[1]*x + th[0] - y)) / m
        temp[1] = th[1] - alpha * (sum((th[1]*x + th[0] - y)*x)) / m
        th[0], th[1] = temp[0], temp[1]
        cos.append(cost(th, x, y))
    return th, cos
```

### 2. **Regularization Journey**
- **Lasso (L1)**: Forces coefficients to exactly zero → feature selection
- **Ridge (L2)**: Shrinks coefficients but keeps all features
- **Elastic Net**: Best of both worlds — `α*L1 + (1-α)*L2`

### 3. **K-Means Clustering**
- Implemented centroid initialization, assignment, and update steps
- Explored elbow method for optimal K

### 4. **XOR Problem**
- Demonstrated why linear models fail on non-linearly separable data
- First exposure to neural network concepts

---

## 📁 Repository Structure

```
My-early-days-of-machine-learning/
├── README.md                                    # This file
├── my first linear regression.ipynb             # Simple linear regression
├── my multiple linear regression.ipynb          # Multiple regression
├── my lasso.ipynb                               # L1 regularization
├── elastic.ipynb                                # Elastic Net
├── polynomial.ipynb                             # Polynomial regression
├── Kmeans.ipynb                                 # K-means clustering
├── NaiveBayes.py                                # Naive Bayes classifier
├── Xor.py                                       # XOR neural network
├── Bioinformatics_String_Matching_assignment.ipynb
├── Gradient_descent_notes/                      # Math derivations
└── ML_Notes/                                    # Study materials (40+ files)
    ├── Linear Regression and Hypothesis Testing.pdf
    ├── Regularization.pdf
    ├── Decision-tree-ch4.pdf
    ├── PCA_CURAJ.pptx
    ├── Random_Graph_CURAJ_3.pptx
    ├── tf-idf.pdf
    ├── HMM.pdf
    ├── ... and many more
```

---

## 🛠️ How to Run

### Prerequisites
```bash
Python 3.6+
Jupyter Notebook
```

### Required Packages (approximate versions from 2018)
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Launch Notebooks
```bash
cd My-early-days-of-machine-learning
jupyter notebook
```

---

## 🎓 What I Learned (Then vs Now)

| Concept | 2018 Understanding | Current Understanding |
|---------|-------------------|----------------------|
| **Gradient Descent** | Manual loops, hardcoded learning rates | Vectorized, adaptive optimizers (Adam, RMSprop) |
| **Regularization** | Just L1/L2 penalties | Elastic Net, Group Lasso, Dropout, Weight Decay |
| **Feature Engineering** | Manual polynomial features | Automated (FeatureTools, AutoML), Embeddings |
| **Validation** | Train/test split only | Cross-validation, Bootstrap, Nested CV |
| **Code Quality** | Hardcoded paths, global variables | Modular, tested, documented, CI/CD |

---

## 📚 Study Materials (ML_Notes/)

The `ML_Notes/` directory contains **40+ curated resources** from my coursework at CURAJ:

### Core ML Theory
- Linear Regression, Logistic Regression, Hypothesis Testing
- Regularization (Lasso, Ridge, Elastic Net)
- Decision Trees, Random Forests, Boosting, Bagging
- Clustering (K-Means, Hierarchical, DBSCAN)
- Dimensionality Reduction (PCA, SVD, Random Projections)

### Advanced Topics
- Deep Learning (CNNs, RNNs, Transformers)
- Graphical Models, HMMs, Bayesian Networks
- Random Graphs, Random Walks, Markov Chains
- TF-IDF, NLP fundamentals
- High-dimensional spaces, Law of Large Numbers

---

## 🕰️ Historical Context

- **Year**: 2018
- **Course**: M.Sc. Computer Science (Big Data Analytics) at Central University of Rajasthan
- **Environment**: Jupyter Notebooks on local machine, scikit-learn 0.19/0.20
- **Hardware**: CPU-only, no GPUs

---

## 🤝 Why Keep This Public?

1. **Documentation of growth** — Shows the learning curve from beginner to practitioner
2. **Educational value** — Others can see "how not to do it" and learn from mistakes
3. **Nostalgia** — Reminds me where I started
4. **Transparency** — Real learning isn't clean or perfect

---

## 👤 Author

**Pradyumna Kumar Sahoo**  
*From "naive attempts" (2018) to production ML systems (2024+)*

- GitHub: [@Prady029](https://github.com/Prady029)
- Portfolio: [prady029.github.io](https://prady029.github.io)
- LinkedIn: [prady029](https://linkedin.com/in/prady029)

---

## 📜 License

MIT License — Feel free to explore, learn, and smile at the beginner code.

---

> *"The expert in anything was once a beginner. The master has failed more times than the beginner has tried."* — Helen Hayes

*Last updated: October 2024 (README only — code preserved as-is from 2018)*