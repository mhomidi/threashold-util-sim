import os
import config

from modules.applications import Application
from modules.applications.markov import MarkovApplication
from modules.applications.queue import QueueApplication

from utils.distribution import *

from modules.scheduler import Scheduler
from modules.scheduler.pref_sched.most_token_first import MostTokenFirstScheduler
from modules.scheduler.pref_sched.baseline.dyn_prop_sp import DynamicProportionalSpaceSlicingScheduler
from modules.scheduler.pref_sched.baseline.fix_prop_sp import FixedProportionalSpaceSlicingScheduler
from modules.scheduler.pref_sched.baseline.lottory_sp import LottorySpaceSlicingScheduler
from modules.scheduler.pref_sched.baseline.lottory_time import LottoryTimeSlicingScheduler
from modules.scheduler.pref_sched.baseline.rr_sp import RoundRobinSpaceSlicingScheduler
from modules.scheduler.pref_sched.baseline.rr_time import RoundRobinTimeSlicingScheduler
from modules.scheduler.queue_sched.baseline.finish_time_fairness import FinishTimeFairnessScheduler

from modules.policies import Policy
from modules.policies.fixed_threshold import FixedThresholdPolicy
from modules.policies.actor_critic import ActorCriticPolicy
from modules.policies import ACPolicy, GandivaFairPolicy

from modules.agents import Agent

json_path = os.path.dirname(os.path.abspath(__file__)) + '/../test/json'


def get_policy(**kwargs) -> Policy:
    policy_type = kwargs.get('type', 'ac')
    budget = kwargs.get('budget', config.get('budget'))
    thr = kwargs.get('thr', 0.0)

    policy = None
    if policy_type == 'ac':
        policy = ACPolicy()
    elif policy_type == 'fixed_thr':
        policy = FixedThresholdPolicy(threshold=thr)
    elif policy_type == 'gandiva':
        policy = GandivaFairPolicy()
    else:
        raise Exception('Policy type \'' + policy_type +
                        '\' is not valid application')
    return policy


def get_application(**kwargs) -> Application:
    app_type = kwargs.get('type', 'markov')
    markov_init_file = kwargs.get(
        'markov_json_path', json_path + "/agents/app1.json")
    piosson_lambda = kwargs.get('p_lambda', 3)

    app = None
    if app_type == 'markov':
        app = MarkovApplication()
        app.init_from_json(markov_init_file)
    elif app_type == 'queue':
        app = QueueApplication()
    else:
        raise Exception('Application type \'' + app_type +
                        '\' is not valid application')
    return app


def get_scheduler(**kwargs) -> Scheduler:
    sched_type = kwargs.get('type', 'mtf')
    n = kwargs.get('n', int(config.get('default_agent_num')))

    sched = None
    if sched_type == 'mtf':
        sched = MostTokenFirstScheduler()
    elif sched_type == 'lottery':
        sched = LottorySpaceSlicingScheduler(n)
    elif sched_type == 'rr':
        sched = RoundRobinSpaceSlicingScheduler(n)
    elif sched_type == 'ftf':
        sched = FinishTimeFairnessScheduler()
    else:
        raise Exception('Scheduler type \'' + sched_type +
                        '\' is not valid application')
    return sched

