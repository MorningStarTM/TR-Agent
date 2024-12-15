import yaml

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to Config objects
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(data)