# ARB4NID

**Adversarial Robustness Benchmark for Network Intrusion Detection**

ARB4NID is a tool for evaluating the robustness of Machine Learning models used in Network Intrusion Detection Systems (NIDS) against adversarial attacks.

---

## Prerequisites

ARB4NID requires **Python 3.11**. Check whether you already have it:

```bash
python3.11 --version
```

If the command is not found, install Python 3.11 first:

- **Recommended — [pyenv](https://github.com/pyenv/pyenv)** (installs Python 3.11 without affecting your system Python):
  ```bash
  # macOS (Homebrew):
  brew install pyenv

  # Linux:
  curl https://pyenv.run | bash

  # then, on any OS:
  pyenv install 3.11.9
  ```

- **Alternative** — download the installer directly from [python.org/downloads](https://www.python.org/downloads/) and install Python 3.11.

You also need **Git** ([git-scm.com](https://git-scm.com/downloads)) to clone the repository, or you can download the project as a ZIP from the GitHub page.

---

## Getting Started

Follow these steps to run the project locally.

**1. Clone the repository**

```bash
git clone https://github.com/antoniosantos1602/ARB4NID.git
```

**2. Create and activate a virtual environment** (Python 3.11)

```bash
pyenv local 3.11.9

python3.11 -m venv venv
source venv/bin/activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
python app.py
```

## Screenshots

| Dataset & Model Selection | Attack Configuration | Results Dashboard |
|:---:|:---:|:---:|
| ![Selection](docs/select_data.png) | ![Configuration](docs/configs.png) | ![Results](docs/results.png) |

---

## Usage

1. **Submit the inputs** — upload a test dataset (CSV) and one or more pre-trained models in `.joblib` format.
2. **Automatic analysis** — ARB4NID inspects the dataset and discovers the domain realism constraints (binary/integer features, dependencies, and statistical references).
3. **Configure the attacks** — choose the attack methods, the mode (untargeted / targeted), and adjust the parameters (if you want).
4. **Analyse & export** — inspect the interactive dashboards and export per-model reports and adversarial datasets (CSV).

### Input format

- **Dataset:** a CSV file.
- **Model:** a trained model stored in `.joblib` format. If normalization was used during training, store the scaler together with the model so the same preprocessing is reproduced.
- Multiple models can be evaluated in the same run.

---

## Available Attacks

ARB4NID integrates a range of adversarial attacks from the ART and Foolbox libraries and the A2PM method, covering white-box, gray-box, and black-box scenarios.

| Attack Method | Knowledge | Objective |
|:---|:---:|:---:|
| FGM | White-Box | Targeted / Untargeted |
| PGD | White-Box | Targeted / Untargeted |
| JSMA | White-Box | Targeted / Untargeted |
| C&W | White-Box | Targeted / Untargeted |
| DeepFool | White-Box | Untargeted |
| ZOO | Black-Box | Targeted / Untargeted |
| Boundary Attack | Black-Box | Targeted / Untargeted |
| HopSkipJump | Black-Box | Targeted / Untargeted |
| A2PM | Gray-Box | Targeted / Untargeted |

---

## Evaluation Metrics

- **Adversarial Accuracy (AA)**
- **Attack Success Rate (ASR)**
- **Attack Reputation Score (ARS)**
- **Realistic Adversarial Accuracy (RAA)**

---

## Robustness Evaluation

Robustness is measured through the **RAA**: higher values mean higher robustness. Because an attacker only needs one effective strategy while a defender must resist them all, two indicators are considered jointly:

- **RAA Mean** — the average RAA across all evaluated attacks (overall behaviour).
- **RAA Min** — the lowest RAA among the attacks (worst-case, revealing specific weaknesses).

Each attack's RAA maps to a robustness level:

| RAA (%) | Robustness Level |
|:---:|:---:|
| 90 – 100 | Very High |
| 70 – 90 | High |
| 50 – 70 | Moderate |
| < 50 | Low |

Combining **RAA Mean** and **RAA Min** (using 70% for the mean and 50% for the minimum), each model is classified into one of four profiles:

| Profile | Condition |
|:---|:---|
| **Highly Robust** | Mean ≥ 70% and Min ≥ 50% |
| **Robust but Vulnerable** | Mean ≥ 70% and Min < 50% |
| **Moderately Robust** | Mean < 70% and Min ≥ 50% |
| **Low Robustness** | Mean < 70% and Min < 50% |

---
