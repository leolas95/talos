from .alert import alert
from .snapshot import snapshot
from .push_notification import push_notification

_ACTIONS = {
    'alert': alert,
    'snapshot': snapshot,
    'push_notification': push_notification
}

ACTIONS_THAT_REQUIRE_FRAME = ['snapshot', 'push_notification']

def do(action_name, *action_args):
    _ACTIONS.get(action_name)(*action_args)
