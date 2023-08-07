import os
import utils


SYS_CONF_FILE = os.path.dirname(os.path.abspath(__file__)) + "/sys_config.json"
MODEL_CONF_FILE = os.path.dirname(os.path.abspath(__file__)) + "/model_config.json"

init_done = False

sys_data: dict = None
model_data: dict = None

def get_data(file_name):
    if file_name == SYS_CONF_FILE and sys_data is not None:
        return sys_data
    if file_name == MODEL_CONF_FILE and model_data is not None:
        return model_data
    return utils.get_json_data_from_file(file_name)

def get(key):
    global init_done
    if not init_done:
        global sys_data
        sys_data = get_data(SYS_CONF_FILE)
        global model_data
        model_data = get_data(MODEL_CONF_FILE)
        init_done = True

    if key in model_data.keys():
        return model_data[key]
    if key in sys_data.keys():
        return sys_data[key]
    raise Exception("Key is not true")