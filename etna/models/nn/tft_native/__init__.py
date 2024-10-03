from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.tft_native.tft import TFTNativeModel
