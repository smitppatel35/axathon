import json
from datetime import datetime

from load_and_predict_model import XGBoost_predict


def _handle_authorities_contacted(_form_data):
    response = {
        "authorities_contacted_ambulance": [0],
        "authorities_contacted_fire": [0],
        "authorities_contacted_none": [0],
        "authorities_contacted_police": [0],
    }

    _authority = "authorities_contacted_" + _form_data['authorities_contacted']
    if response.__contains__(_authority):
        response[_authority][0] = 1
        _remove_item_from_form_data(_form_data, ['authorities_contacted'])
        return response
    else:
        raise Exception("Invalid Feature: authorities_contacted passed to model")


def _handle_collision_type(_form_data):
    response = {
        "collision_type_front": [0],
        "collision_type_rear": [0],
        "collision_type_side": [0],
        "collision_type_na": [0],
    }

    _collision_type = "collision_type_" + _form_data['collision_type']
    if response.__contains__(_collision_type):
        response[_collision_type][0] = 1
        _remove_item_from_form_data(_form_data, ['collision_type'])

        return response
    else:
        raise Exception("Invalid Feature: collision_type passed to model")


def _remove_item_from_form_data(_form_data: dict, _items):
    [_form_data.pop(_item) for _item in _items]


def _extract_incident_dates(_form_data):
    # 2023-05-11T11: 05
    format_str = '%Y-%m-%dT%H:%M'
    date_obj = datetime.strptime(_form_data['incident_date'], format_str)

    _remove_item_from_form_data(_form_data, ['incident_date'])

    return {
        "incident_day": [date_obj.day],
        "incident_month": [date_obj.month],
        "incident_hour": [date_obj.hour]
    }


def _handle_customer_gender(_form_data):
    response = {
        "customer_gender_male": [0],
        "customer_gender_female": [0],
    }

    _authority = "customer_gender_" + _form_data['customer_gender']
    if response.__contains__(_authority):
        response[_authority][0] = 1
        _remove_item_from_form_data(_form_data, ['customer_gender'])
        return response
    else:
        raise Exception("Invalid Feature: customer_gender passed to model")


def _handle_driver_relationship(_form_data):
    response = {
        "driver_relationship_child": [0],
        "driver_relationship_na": [0],
        "driver_relationship_other": [0],
        "driver_relationship_self": [0],
        "driver_relationship_spouse": [0]
    }

    _authority = "driver_relationship_" + _form_data['driver_relationship']
    if response.__contains__(_authority):
        response[_authority] = [1]
        _remove_item_from_form_data(_form_data, ['driver_relationship'])
        return response
    else:
        raise Exception("Invalid Feature: driver_relationship passed to model")


def _handle_incident_type(_form_data):
    response = {
        "incident_type_breakin": [0],
        "incident_type_collision": [0],
        "incident_type_theft": [0]
    }

    _authority = "incident_type_" + _form_data['incident_type']
    if response.__contains__(_authority):
        response[_authority][0] = 1
        _remove_item_from_form_data(_form_data, ['incident_type'])
        return response
    else:
        raise Exception("Invalid Feature: incident_type passed to model")


def _handle_policy_state(_form_data):
    response = {
        "policy_state_az": [0],
        "policy_state_ca": [0],
        "policy_state_id": [0],
        "policy_state_nv": [0],
        "policy_state_or": [0],
        "policy_state_wa": [0],
    }

    _authority = "policy_state_" + _form_data['policy_state']
    if response.__contains__(_authority):
        response[_authority][0] = 1
        _remove_item_from_form_data(_form_data, ['policy_state'])
        return response
    else:
        raise Exception("Invalid Feature: policy_state passed to model")


def pre_process_form_data(_model_input, _form_data):
    _model_input.update(_extract_incident_dates(_form_data))
    _model_input.update(_handle_authorities_contacted(_form_data))
    _model_input.update(_handle_collision_type(_form_data))
    _model_input.update(_handle_customer_gender(_form_data))
    _model_input.update(_handle_driver_relationship(_form_data))
    _model_input.update(_handle_incident_type(_form_data))
    _model_input.update(_handle_policy_state(_form_data))

    _remove_item_from_form_data(_form_data, ['customer_education'])

    # Convert JSON string to int64 and float64
    for _item_from_form in _form_data:

        if _item_from_form != "customer_education":
            if _item_from_form == "police_report_available":
                _form_data[_item_from_form] = [1] if _form_data[_item_from_form] == 'Yes' else [0]
            else:
                if str(_form_data[_item_from_form]).__contains__("."):
                    _form_data[_item_from_form] = [float(_form_data[_item_from_form])]
                else:
                    _form_data[_item_from_form] = [int(_form_data[_item_from_form])]

    _model_input.update(_form_data)


def predict(_form_data):
    _model_input = {}
    pre_process_form_data(_model_input, _form_data)

    result = XGBoost_predict(_model_input)

    return result
