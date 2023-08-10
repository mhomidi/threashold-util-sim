import os
import json
import optuna
import pandas as pd
import subprocess

path = os.path.dirname(os.path.abspath(__file__))
config_file = path + "/config/model_config.json"

def get_util_mean() -> float:
    data = pd.read_csv(path + '/report_utils.csv')
    return float(data.mean().mean())


def objective(trial: optuna.trial.Trial) -> float:
    model_conf = dict()
    alpha = trial.suggest_categorical("alpha", [2, 4, 6, 8, 10])
    beta = trial.suggest_categorical("beta", [1.5, 2.0, 2.5, 3.0])

    model_conf['l1_in'] = trial.suggest_categorical('l1_in', [64, 128, 192, 256])

    model_conf['actor_l2_in'] = trial.suggest_categorical('actor_l2_in', [64, 128, 192, 256])
    model_conf['actor_l3_in'] = trial.suggest_categorical('actor_l3_in', [64, 128, 192, 256])
    model_conf['actor_l4_in'] = trial.suggest_categorical('actor_l4_in', [64, 128, 192, 256])

    model_conf['critic_l2_in'] = trial.suggest_categorical('critic_l2_in', [64, 128, 192, 256])
    model_conf['critic_l3_in'] = trial.suggest_categorical('critic_l3_in', [64, 128, 192, 256])
    model_conf['critic_l4_in'] = trial.suggest_categorical('critic_l4_in', [64, 128, 192, 256])

    model_conf['actor_lr'] = trial.suggest_categorical("actor_lr", [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4])
    model_conf['critic_lr'] = alpha * model_conf['actor_lr']

    model_conf['decay_factor'] = 1 - beta * model_conf['actor_lr']
    with open(config_file, "w") as file:
        json.dump(model_conf, file)

    subprocess.run(['python3', path + '/test/ac_policy.py'], check=True)

    return get_util_mean()


if __name__ == "__main__":
    print(config_file)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)
    print("################## BEST DATA ##############")
    print("Best value:", study.best_value)
    print("################## BEST PARAM #############")
    print("param:", study.best_params)
