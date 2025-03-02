from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepstate.linear_dynamic_system import LDS
    from etna.models.nn.deepstate.state_space_model import CompositeSSM
    from etna.models.nn.deepstate.state_space_model import DailySeasonalitySSM
    from etna.models.nn.deepstate.state_space_model import LevelSSM
    from etna.models.nn.deepstate.state_space_model import LevelTrendSSM
    from etna.models.nn.deepstate.state_space_model import SeasonalitySSM
    from etna.models.nn.deepstate.state_space_model import WeeklySeasonalitySSM
    from etna.models.nn.deepstate.state_space_model import YearlySeasonalitySSM
