from .alert import alert
from .snapshot import snapshot

_ACTIONS = {
    'alert': alert,
    'snapshot': snapshot
}

ACTIONS_THAT_REQUIRE_FRAME = ['snapshot']

def do(action_name, *action_args):
    _ACTIONS.get(action_name)(*action_args)
