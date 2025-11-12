## Top Activations

Placeholder for the “max-activation image grid” experiment. The inputs/outputs are the same ones referenced in the talk (scan an `ImageFolder`, keep the top-k activations for a chosen layer/channel, and tile the winners).

### Status
Implementation is still being ported. Expected pieces:
- `top_activations.py` – CLI that walks a dataset, runs the backbone, and writes the resulting grid.
- Sample configs for GoogLeNet/Inception v1 plus alternative datasets (Imagenette, custom folders).

### How to Help
- Drop any prototype script into this directory (feel free to start from torchvision’s activation hooks).
- Document the dataset path you used so others can replay the ranking.

Until the script lands, check the feature-visualization README for Lucent-based neuron introspection examples.
