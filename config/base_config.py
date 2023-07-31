# create copy from this file with "config.py" name
# in this directory

CLUSTERS_NUM = 10
DIST_SAMPLE = 10
EPSILON = 1e-3
BUDGET = 10
DEFAULT_NUM_AGENT = 5

THRESHOLDS_NUM = 10
DISCOUNT_FACTOR = 0.99

AC_EPISODES = 3e3
NR_ROUNDS = 1e3

DEFAULT_STATE_VALUE = 0.0

BUDGET_BIN = 4  # Code of no_regret_policy_agent.py -> get_budget_index related to this number (DO NOT CHANGE)
POLICY_NUM = BUDGET_BIN * THRESHOLDS_NUM
