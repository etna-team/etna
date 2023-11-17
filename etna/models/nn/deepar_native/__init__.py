from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepar_native.deepar import DeepARNativeModel
