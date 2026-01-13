# Interpretability & Uncertainty in Neural Networks

This project explores how neural network classifiers behave under input corruption,
and why confidence alone is not a reliable indicator of correctness.

The experiments focus on:
- Saliency-based interpretability
- Confidence vs correctness under noise
- Epistemic uncertainty using Monte Carlo Dropout
- Rejection (abstention) mechanisms

## Motivation

Modern neural networks are often highly confident, even when they are wrong —
especially when inputs differ from the training distribution.

Understanding this behavior is critical for safety-sensitive applications such as:
- Autonomous driving
- Robotics
- Medical AI
- Aerospace systems

## Experiments

### 1. Saliency Maps
Visualized gradient-based saliency maps to inspect which pixels influence predictions.
Observed stable explanations for clean inputs and scattered explanations under heavy noise.

### 2. Noise Sweep
Added increasing levels of Gaussian noise to MNIST images and measured:
- Prediction
- Softmax confidence
- Failure modes

Found non-monotonic behavior:
moderate noise caused confident misclassification, while higher noise sometimes restored correct predictions.

### 3. Confidence Thresholding
Implemented a confidence-based reject option.
Result: confidence thresholds alone failed to detect many high-confidence errors.

### 4. Monte Carlo Dropout
Used stochastic forward passes at inference time to estimate epistemic uncertainty.
Observed:
- Low uncertainty for clean inputs
- Gradual increase in uncertainty with noise
- Better signal for rejection than softmax confidence

## Key Takeaways

- Softmax confidence ≠ reliability
- Neural networks are forced decision-makers
- Noise does not increase uncertainty monotonically
- Uncertainty estimation improves safety but is not sufficient alone
- Reject options must be explicitly designed

## Limitations

- MNIST is a simple dataset
- Random noise is not adversarial
- Models were intentionally small for interpretability

Future work includes:
- CIFAR-10 experiments
- Adversarial perturbations
- Ensemble-based uncertainty
- Explicit "unknown" classes

## Technologies Used

- Python
- PyTorch
- Captum
- Matplotlib

## Author
Amaar A.

## Purpose
Exploratory research project focused on AI interpretability and uncertainty estimation.
