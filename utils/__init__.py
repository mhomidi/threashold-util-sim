

import json


def get_json_data_from_file(file_name: str):
    with open(file_name) as file:
        return json.load(file)

