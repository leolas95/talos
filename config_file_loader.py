import pathlib

def load(filename):
    # Get extension without the leading dot
    extension = pathlib.Path(filename).suffix[1:]

    if extension == 'json':
        import json
        with open(filename, 'r') as file:
            result = json.load(file)

    elif extension in ('yaml', 'yml'):
        import yaml
        with open(filename, 'r') as file:
            result = yaml.load(file, Loader=yaml.Loader)

    else:
        import xmltodict
        # Use json to transform the OrderedDict returned from xmltodict.parse to a normal dict,
        # not a difference but ok...
        import json
        with open(filename, 'r') as file:
            result = xmltodict.parse(file.read())
            result = dict(result['root'])
            result = json.loads(json.dumps(result))

            # xmltodict includes some garbage when parsing lists, so we still have to
            # manually parse input (sigh...)

            # If there are targets with empty properties, replace None value with {}
            if result.get('targets'):
                for target, targetval in result['targets'].items():
                    if targetval is None:
                        result['targets'][target] = {}

            # If there are activities with empty values, replace None with {}
            if result.get('activities'):
                for activity in result['activities'].keys():
                    if result['activities'][activity] is None:
                        result['activities'][activity] = {}

            # Transform objects with 'item' key to a list of those objects
            if result.get('activities_conditions'):
                result['activities_conditions'] = result['activities_conditions']['item']

                if type(result['activities_conditions']).__name__ != 'list':
                    result['activities_conditions'] = [
                        result['activities_conditions']]

                for condition in result['activities_conditions']:
                    if type(condition['action_arguments']['item']).__name__ != 'list':
                        condition['action_arguments'] = [
                            condition['action_arguments']['item']]

            if result.get('targets_conditions'):
                result['targets_conditions'] = result['targets_conditions']['item']

                if type(result['targets_conditions']).__name__ != 'list':
                    result['targets_conditions'] = [
                        result['targets_conditions']]

                for condition in result['targets_conditions']:
                    if type(condition['action_arguments']['item']).__name__ != 'list':
                        condition['action_arguments'] = [
                            condition['action_arguments']['item']]

    return result
