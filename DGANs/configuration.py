import yaml
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


config = AttrDict.from_nested_dicts(
    yaml.load(open(os.path.join(os.path.dirname(__file__), "configuration.yaml"), "r"), Loader=yaml.FullLoader))
