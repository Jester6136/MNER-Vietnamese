import math
import torch.nn as nn

def reinitialize_conv2d(model):
    """
    Reinitialize weights and biases of all nn.Conv2d layers in the model.
    Weights: Kaiming uniform initialization (suitable for ReLU-like non-linearities).
    Biases: Constant initialization to 0.0.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Initialize weights with Kaiming uniform
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # a=sqrt(5) for ReLU-like non-linearities
            # Initialize biases to 0.0 if they exist
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            # nn.initialize batch norm parameters (if used)
            nn.init.constant_(module.weight, 1.0)  # Gamma
            nn.init.constant_(module.bias, 0.0)   # Beta
            nn.init.constant_(module.running_mean, 0.0)
            nn.init.constant_(module.running_var, 1.0)

def safe_init_weights(m):
    """Init an toàn cho các layer custom (Conv2d, Linear, CRF)."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif hasattr(m, "reset_parameters") and (
        m.__class__.__name__ in ["CRF"]
    ):
        m.reset_parameters()


def reinit_custom_modules(model):
    """Chạy init cho tất cả module custom trong UMT_PixelCNN."""
    # Init image decoder
    if hasattr(model, "image_decoder"):
        model.image_decoder.apply(safe_init_weights)

    # Init aux classifier
    if hasattr(model, "aux_classifier"):
        model.aux_classifier.apply(safe_init_weights)

    # Init CRFs
    if hasattr(model, "crf"):
        model.crf.reset_parameters()
    if hasattr(model, "aux_crf"):
        model.aux_crf.reset_parameters()

    print("✅ Custom modules re-initialized safely (PhoBERT untouched).")
