from baselines.greedy_extractor import policy as greedy_extractor_policy, BASELINE_NAME as GREEDY_NAME
from baselines.conservative_rotation import policy as conservative_rotation_policy, BASELINE_NAME as CONSERVATIVE_NAME
from baselines.weather_aware_rotation import policy as weather_aware_rotation_policy, BASELINE_NAME as WEATHER_AWARE_NAME

BASELINES = {
    GREEDY_NAME:       greedy_extractor_policy,
    CONSERVATIVE_NAME: conservative_rotation_policy,
    WEATHER_AWARE_NAME: weather_aware_rotation_policy,
}
