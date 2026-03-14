# DEMANDE: Density Matrix Neural Density Estimation

**DEMANDE** (**D**ensity **M**atrix **NE**ural **D**ensity **E**stimation) is a neural density estimation method based on **density matrices** and **adaptive Fourier features**. It provides a flexible machine-learning approach to estimate probability density functions from data, grounded in the mathematical formalism of density matrices commonly used in quantum mechanics. ([Semantic Scholar][1])

---

## 📌 Overview

Traditional density estimation methods like Kernel Density Estimation (KDE) scale poorly with dimensionality and dataset size. DEMANDE models densities using density matrices combined with adaptive Fourier feature maps, yielding a scalable, data-driven estimator that can be integrated with deep learning tools and evaluated efficiently. ([Semantic Scholar][1])

---

## 🔍 Features

* **Density matrix representation** of probability distributions
* **Adaptive Fourier features** to approximate kernels and embed data
* Python implementation with example notebooks
* Modular and extensible for research and experiments
* Includes tests and baseline usage examples

---

## 🚀 Getting Started


# Setup


## Installation

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the `learning_with_density_matrices` environment:

```bash
conda env create -f environment.yml

conda activate learning-with-density-matrices
```

Next, install the package:

```bash
pip install -e .
```

or if you want development dependencies as well:

```bash
pip install -e .[dev]
```


## Gitsubmodules update

This repository rely on some gitsubmodules. To update them run:

```bash
git submodule update --init --recursive
```

# Create Directories

- `mkdir reports mlflow data` 

## Ml-flow

All the experiments will be saved on Ml-flow in the following path using `sqlite`: `mlflow/`

```bash
mkdir mlflow/
```

After running your experiments, you can launch the ml-flow dashboard by running the following command:

```bash
mlflow ui --port 8080 --backend-store-uri sqlite:///mlflow/tracking.db
```

---

## 📁 Repository Structure

```
📦 demande
├── src/                    # Core implementation
├── notebooks/             # Python scripts to run demos
├── tests/                  # Unit and integration tests
├── data/                   # Example dataset
├── pyproject.toml
├── README.md
└── LICENSE
```

---



---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

---

## 📚 Related Work

DEMANDE builds on the idea of density matrices as probability density estimators, with roots in kernel and random Fourier feature methods. ([Semantic Scholar][1])

---

## 📄 Citation

If you use this code in your research, please cite:

```
@article{gallego2023demande,
  title={DEMANDE: Density Matrix Neural Density Estimation},
  author={Gallego-Meji{\'a}, Joseph A. and Gonz{\'a}lez, Fabio A.},
  journal={IEEE Access},
  year={2023}
}
```

(This is a placeholder citation — adjust to the official published version and BibTeX entry.) ([ResearchGate][4])

---

## 🧑‍🔬 Contributing

Contributions are welcome! Please open issues for bugs, feature requests, or improvements.

1. Fork the repo
2. Create a feature branch
3. Add tests for new behavior
4. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

If you want, I can also generate badges (e.g., build status, PyPI, citations) or add **installation via pip/Conda** and **API reference** sections to the README.

[1]: https://www.semanticscholar.org/paper/DEMANDE%3A-Density-Matrix-Neural-Density-Estimation-Gallego-Mejia-Gonz%C3%A1lez/1a08e8c5607646eab2d1cc8590171cfe0b6dcf6f?utm_source=chatgpt.com "[PDF] DEMANDE: Density Matrix Neural Density Estimation"
[2]: https://paperswithcode.com/paper/fast-kernel-density-estimation-with-density?utm_source=chatgpt.com "Fast Kernel Density Estimation with Density Matrices and Random Fourier Features | Papers With Code"
[3]: https://link.springer.com/article/10.1007/s42484-022-00079-9?utm_source=chatgpt.com "Learning with density matrices and random features | Quantum Machine Intelligence | Springer Nature Link"
[4]: https://www.researchgate.net/publication/371001879_DEMANDE_Density_Matrix_Neural_Density_Estimation?utm_source=chatgpt.com "(PDF) DEMANDE: Density Matrix Neural Density Estimation"
