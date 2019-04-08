import cv2
import actions.actions as actions

def handle_activity(activity, activities_conditions, frame):
    cv2.putText(frame, activity.capitalize(), (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    if activities_conditions is not None:
        conditions = activities_conditions[:]
        for condition in conditions:
            if activity != condition['activity']:
                continue

            action = condition['action']
            action_args = condition['action_arguments']

            if action in actions.ACTIONS_THAT_REQUIRE_FRAME:
                action_args.append(frame)

            actions.do(action, *action_args)
            activities_conditions.remove(condition)
