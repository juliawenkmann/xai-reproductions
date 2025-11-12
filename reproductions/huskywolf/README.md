# Husky vs. Wolf – Ribeiro 2016


**1) Load the model (or *train if missing*) in one line**

```python
from huskywolf import get_model
model = get_model(  # loads if weights exist, otherwise trains and saves
    weights_path="models/model_weights/binary_classifier.pth",
    data_root="./",        # used only if training is needed
    train_if_missing=True, # default
    epochs=20, lr=0.01
)
```

**2) Predict on an image**

```python
from huskywolf import predict_image
probs = predict_image("my_image.jpg", weights_path="model_weights/binary_classifier.pth")
print(probs)  # {'prob_class1_wolf': 0.73, 'prob_class0_husky': 0.27}
```

**3) Train from scratch (explicit)**

```bash
python scripts/train.py --data . --out model_weights/binary_classifier.pth --epochs 20 --lr 0.01
```

**4) Make a Grad-CAM explanation**

```bash
pip install grad-cam imageio
python scripts/explain.py --img my_image.jpg --weights model_weights/binary_classifier.pth --out cam.jpg
```

