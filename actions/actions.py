from .alert import alert

_ACTIONS = {
    'alert': alert
}

def do(action_name, *action_args):
    _ACTIONS.get(action_name)(*action_args)
