
from config import config
from modules.policies import Policy


class BudgetThresholdPolicy(Policy):

    def get_thresholds(self) -> list:
        thr = list()
        idx = self.index
        for i in range(config.BUDGET_BIN):
            if idx > 0:
                thr.append((idx % config.THRESHOLDS_NUM) /
                           float(config.THRESHOLDS_NUM))
            else:
                thr.append(0.)
            idx / config.THRESHOLDS_NUM
        return thr
