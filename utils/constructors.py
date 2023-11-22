import os
import config

from modules.applications.markov import MarkovApplication
from modules.applications.distribution import DistributionApplication
from utils.distribution import *

from modules.scheduler.most_token_first import MostTokenFirstScheduler
from modules.scheduler.dyn_prop_sp import DynamicProportionalSpaceSlicingScheduler
from modules.scheduler.fix_prop_sp import FixedProportionalSpaceSlicingScheduler
from modules.scheduler.lottory_sp import LottorySpaceSlicingScheduler
from modules.scheduler.lottory_time import LottoryTimeSlicingScheduler
from modules.scheduler.rr_sp import RoundRobinSpaceSlicingScheduler
from modules.scheduler.rr_time import RoundRobinTimeSlicingScheduler

from modules.policies.fixed_threshold import FixedThresholdPolicy
from modules.policies.actor_critic import ActorCriticPolicy

json_path = os.path.dirname(os.path.abspath(__file__)) + '/../test/json'

def get_policy(**kwargs):
    policy_type = kwargs.get('type', 'ac')
    budget = kwargs.get('budget', config.get('budget'))
    thr = kwargs.get('thr', 0.0)

    policy = None
    if policy_type == 'ac':
        policy = ActorCriticPolicy(budget=budget)
    elif policy_type == 'fixed_thr':
        policy = FixedThresholdPolicy(threshold=thr)
    else:
        raise Exception('Policy type \'' + policy_type + '\' is not valid application')
    return policy


def get_application(**kwargs):
    app_type = kwargs.get('type', 'markov')
    markov_init_file = kwargs.get('markov_json_path', json_path + "/agents/app1.json")
    piosson_lambda = kwargs.get('p_lambda', 3)

    app = None
    if app_type == 'markov':
        app = MarkovApplication()
        app.init_from_json(markov_init_file)
    elif app_type == 'poisson':
        app = DistributionApplication(generator=PoissonGenerator(lam=piosson_lambda))
    elif app_type == 'uniform':
        app = DistributionApplication(generator=UniformGenerator())
    else:
        raise Exception('Application type \'' + app_type + '\' is not valid application')
    return app

def get_scheduler(**kwargs):
    sched_type = kwargs.get('type', 'mtf')
    n = kwargs.get('n', int(config.get('default_agent_num')))

    sched = None
    if sched_type == 'mtf':
        sched = MostTokenFirstScheduler()
    elif sched_type == 'lottery':
        sched = LottorySpaceSlicingScheduler(n)
    elif sched_type == 'rr':
        sched = RoundRobinSpaceSlicingScheduler(n)
    else:
        raise Exception('Scheduler type \'' + sched_type + '\' is not valid application')
    return sched