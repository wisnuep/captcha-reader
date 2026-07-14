"""
Definisi arsitektur model CAPTCHA OCR.

Arsitektur ini HARUS identik dengan yang dipakai saat training di notebook,
supaya bobot (checkpoint) hasil training bisa di-load dengan benar.
Class ini sengaja dibuat sebagai nn.Module biasa (bukan LightningModule)
supaya saat deploy tidak perlu install pytorch-lightning (lebih ringan & cepat).
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Jumlah karakter dalam satu captcha
CAPTCHA_LENGTH = 6
# Jumlah kemungkinan karakter berbeda
NUM_CHARACTERS = 21

# Harus SAMA PERSIS dengan encoding_dict/decoding_dict di notebook (CaptchaDataset)
DECODING_DICT = {
    0: 'a', 1: 'f', 2: 'e', 3: 'c', 4: 'b', 5: 'h', 6: 'v', 7: 'z',
    8: '2', 9: 'x', 10: 'g', 11: 'm', 12: 'r', 13: 'u', 14: 'p',
    15: 's', 16: 'd', 17: 'n', 18: '6', 19: 'k', 20: 't',
}

# Mean & std precomputed dari training set (dipakai saat normalisasi gambar)
NORM_MEAN = 0.7570
NORM_STD = 0.3110


class CaptchaModel(nn.Module):
    def __init__(self, num_classes=CAPTCHA_LENGTH, num_characters=NUM_CHARACTERS, input_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_characters = num_characters

        self.resnet50 = models.resnet50()
        self.resnet50.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=self.num_characters * self.num_classes, bias=True),
        )

    def forward(self, x):
        return self.resnet50(x)


def load_model(checkpoint_path, device="cpu"):
    """
    Memuat model dari file checkpoint.
    Mendukung:
    - checkpoint PyTorch Lightning (.ckpt) -> berisi key 'state_dict'
    - state_dict biasa hasil torch.save(model.state_dict(), ...) (.pth/.pt)
    """
    model = CaptchaModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Bersihkan prefix "model." jika ada (kadang muncul tergantung cara save)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model
