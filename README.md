# Deep Learning II – PW5: Adversarial Examples & Git Usage

**Course:** DL3AIISO

This repository contains the work for the class exercise on adversarial examples, where the objective is twofold: to learn effective Git project management and to explore how neural networks can be fooled with subtle adversarial perturbations.

---

## Objective

- **Adversarial Examples:** Learn how minimal, often imperceptible, perturbations can degrade the performance of neural networks trained on the CIFAR10 dataset.

---

## Project Overview

The project demonstrates two popular adversarial attack methods:

1. **Fast Gradient Sign Method (FGSM):**
   - **Method:** Computes the perturbation in one step as:
     $$
     \delta = \varepsilon \cdot \text{sign}(\nabla_x L(f(x), y))
     $$
   - **Pros & Cons:** Fast but less effective compared to iterative methods.

2. **Projected Gradient Descent (PGD):**
   - **Method:** Uses an iterative approach:
     $$
     \delta \leftarrow \Pi_{B_p(x, \varepsilon)}\Bigl(\delta + \alpha \cdot \nabla_x L(f(x+\delta), y)\Bigr)
     $$
     where \(\Pi_{B_p(x, \varepsilon)}\) projects the perturbation back into the allowed \(\ell_p\)-ball of radius \(\varepsilon\).
   - **Pros & Cons:** More effective but computationally more intensive.

---


## Repository Structure

The repository includes the following key files and directories:

- **Python Scripts:**
  - `attack_utils.py` – Implements FGSM and PGD attack methods.
  - `evaluate.py` – Script to evaluate the model under adversarial attacks.
  - `models.py` – Contains neural network model definitions (including modifications for CNN and ResNet18).
  - `train.py` – Script to train models on the CIFAR10 dataset.
  - `run_xp.py` – Script to automate multiple experiments with varying hyperparameters.
  - `utils.py` – Utility functions used across the project.
- **Configuration & Environment:**
  - `requirements.txt` and `environment.yml` – Lists of dependencies.
  - `config/` – Additional configuration files.
- **Experiments & Results:**
  - `experiments/` – Folders containing training logs, results, and experiment outputs.
  - Example images and result files (e.g., `examples.png`, `results.png`, `results.txt`).
- **Pre-trained Models:**
  - `model.pth` – Saved model weights.

---

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AlexVerine/TP5.git
   ```
2. **Create and Update Your Environment:**
   - Install Git if you haven't already. Installation instructions can be found at [Git SCM](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
   - Use `conda` or `virtualenv` along with the provided configuration files:
     ```bash
     pip install -r requirements.txt
     ```
   - Alternatively, set up the environment using:
     ```bash
     conda env create -f environment.yml
     ```

---

## Model Training

### Linear Model
- **Training:** Run the training script to start with a simple linear model:
  ```bash
  python train.py
  ```
- **Output:** Check results in the `experiments/base/` folder.

### Convolutional Neural Network (CNN)
- **Modifications:** Update `models.py` to implement a CNN with:
  - 3 convolutional layers (filters: 32, 64, 128) with ReLU and Max pooling.
  - 2 fully connected layers (sizes: 512 and 10).
- **Training:** Execute:
  ```bash
  python train.py --path cnn --model cnn --epochs 10
  ```
- **Output:** Results will be saved in `experiments/cnn/`.

### ResNet18 Fine-Tuning
- **Modifications:** In `models.py`, modify the `ResNet18` class to replace the final fully connected layer (i.e., `self.resnet.fc`) with a linear layer (512 inputs, 10 outputs).
- **Training:** Run:
  ```bash
  python train.py --path resnet18 --model resnet18 --epochs 10
  ```
- **Output:** Check the `experiments/resnet18/` folder for results.

---

## Implementing Adversarial Attacks

### FGSM Attack
- **Implementation:** Complete the `compute` method in the `FastGradientSignMethod` class within `attack_utils.py`.
- **Testing:** Run the evaluation script:
  ```bash
  python evaluate.py --path cnn --model cnn --attack fgsm --epsilon 0.05
  ```
- **Experimentation:** Use `run_xp.py` to test different ε values (e.g., 1e-1, 1e-2, 1e-3).

### PGD Attack
- **Implementation:** Complete the `compute` method in the `ProjectedGradientDescent` class within `attack_utils.py`.
- **Testing:** Evaluate using:
  ```bash
  python evaluate.py --path cnn --model cnn --attack pgd --epsilon 0.1
  ```
- **Hyperparameter Tuning:** For a budget of ε = 0.05, experiment with different numbers of steps and step sizes (recommended: α ≈ ε/T).
- **Enhancement:** Modify the initialization in `attack_utils.py` to automatically set the step size α to ε/T when not provided.

---

## Experimentation and Comparison

- **Attack Comparison:**  
  Compare the effectiveness of FGSM and PGD on:
  - CNN vs. fully connected (FC) models.
  - Fine-tuned ResNet18 versus an overfitted ResNet18.
- **Observations:**  
  Notice the changes in accuracy and robustness of the models when subjected to adversarial attacks.

---

## Conclusion

This project demonstrates how even highly accurate neural networks can be vulnerable to small adversarial perturbations, emphasizing the need for robust defenses.
