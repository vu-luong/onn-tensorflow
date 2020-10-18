# Online Neural Network (ONN)

This is a Tensorflow implementation of the [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705) paper.

## Installing
```

```

## How to use
```python
from benchmark.onn import ONN

model = ONN(n_features, n_classes, n_layers=n_layers, learning_rate=learning_rate)

# Make prediction
pred = model.predict(inputs)

# Update model
model.partial_fit(inputs, targets)

```

* See the file `main.py` for prequentially evaluate ONN+ADWIN

## References
- [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705)
