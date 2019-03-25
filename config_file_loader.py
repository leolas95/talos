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
            if result['targets_conditions'] is None:
                result['targets_conditions'] = []
                
            if result['activities_conditions'] is None:
                result['targets_conditions'] = []

    return result
