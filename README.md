# MARS: Resource Allocation in Multi-Agent System Simulator

MARS serves as a resource allocation simulator for any multi-agent system. It has been designed to simulate the sharing of resources in situations where multiple users compete with each other to utilize the available resources.

## Installation
Installation of MARS is too easy. At it is written in Python, you just need to create a virtual environment with python and install the dependencies:

```zsh
git clone https://github.com/mhomidi/threashold-util-sim.git MARS
cd MARS
virtrualenv .env
source .env/bin/activate

# Do any change you want with sys_config.json but it should be copied
cp config/sys_config_default.json config/sys_config.json


pip install -r requirement.txt
```

## Test
All test are written in `test` directory. Example for running a finish time fairness allocation you just need to run
```zsh
python test/application/test.py --agent queue --app queue --sched ftf
```

You can get the help of the running with:
```zsh
python test.py [-s|--sched] <scheduler> [-p|--policy] <policy> [-a|app] <app_type> [-g|--agent] <agent>
```

* `scheduler`: specifies the type of scheduler. If you have a queue application, then you have to choose `queue` for `agent` and `app`

* `agent`: specifies the type of agent.

* `policy`: specifies the type of policy (fixed_thr, ac, etc.)

* `app`: specifies the type of application
