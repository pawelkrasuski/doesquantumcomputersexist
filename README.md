# Qiskit Machine Learning Tutorials
This repository contains tutorials for learning Qiskit with a focus on machine learning features. These tutorials are designed to help you get started with quantum machine learning using IBM's Qiskit framework.
Prerequisites
Before you begin, ensure you have the following installed:
> Python 3.7 or later
> pip (Python package installer)
## Installation
To set up your environment and install the necessary packages, follow these steps:
Clone this repository:
`text
git clone https://github.com/pawelkrasuski/doesquantumcomputersexist.git
cd doesquantumcomputersexist
`
## Create a virtual environment (optional but recommended):
`text
python -m venv qiskit-env
source qiskit-env/bin/activate  # On Windows, use: qiskit-env\Scripts\activate
`
## Install Qiskit and Qiskit Machine Learning:
`text
pip install qiskit qiskit-machine-learning
`
## Install additional dependencies:
`text
pip install jupyter matplotlib numpy scipy scikit-learn
`
## For optional PyTorch integration:
`text
pip install 'qiskit-machine-learning[torch]'
`
## For optional sparse array support:
`text
pip install 'qiskit-machine-learning[sparse]'
`
