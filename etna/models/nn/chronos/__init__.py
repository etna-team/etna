from etna import SETTINGS

if SETTINGS.chronos_required:
    from etna.models.nn.chronos.chronos import ChronosModel
    from etna.models.nn.chronos.chronos_bolt import ChronosBoltModel
