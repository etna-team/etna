from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepar.deepar import DeepARModel
