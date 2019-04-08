from utils import get_value
import actions.actions as actions

def handle_targets_conditions(targets_conditions, counters, frame):
    # List to keep the conditions that weren't true
    remaining_conditions = []
    for condition in targets_conditions:
        # Get value of left operand
        left_operand = condition['condition']['left_operand']
        left_operand_value = get_value(left_operand, counters)

        operator = condition['condition']['operator']

        # Get value of right operand
        right_operand = condition['condition']['right_operand']
        right_operand_value = get_value(right_operand, counters)

        left_operand = left_operand.replace('-', '_')
        context = {
            left_operand: left_operand_value,
            right_operand: right_operand_value
        }

        expression = f"{left_operand} {operator} {right_operand}"
        expression_value = eval(expression, context)

        if expression_value is True:
            # Execute action
            action = condition['action']
            action_args = condition['action_arguments']
            # If the action requires the frame, append it to the arguments
            if action in actions.ACTIONS_THAT_REQUIRE_FRAME:
                action_args.append(frame)
            actions.do(action, *action_args)
        else:
            # If the current condition wasn't met we keep it for the next round
            remaining_conditions.append(condition)

    # Update the list to the conditions that haven't been met
    targets_conditions[:] = remaining_conditions[:]
