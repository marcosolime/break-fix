# break-fix
### We perform **adversarial attacks** to a CNN trained on **medical images**. Then, we apply counter measures to make the model robust and mantain high accuracy.
The project is part of the course _"Ethics in Artificial Intelligence"_ at the University of Bologna. 

### Contributors:
- Alessandro Folloni
- Daniele Napolitano
- Marco Solime



## Breaking
We perform the following attacks (code in the `sandbox.ipynb` notebook):
- **Data poisoning**
- **Adversarial examples**:
    - **FGSM**: Fast Gradient Sign Method
    - **PGD**: Projected Gradient Descent
- **Biasing the loss**:
    - **NoisyLoss**
    - **FoolingLoss**
- **Manipulating Gradient direction**

## Fixing
We apply the following counter measures:
- **Data Augmentation**: makes the model robust and invariant to small changes in the input
- **Adversarial Training**: to contrast adversarial examples
- **Pattern Detector**: We train a model to detect the possible presence of adversarial examples, to be used in the early stages of the production pipeline

## Requirements
To run the code, you need to install the libraries from the `requirements.txt` file. You can do it by running the following command:
```bash
pip install -r requirements.txt
```