from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class SimpleConfig:
    def merge_with_yaml(self, path):
        with open(path, "r", encoding="utf-8") as file:
            body = ""
            for line in file:
                if line.startswith("# include") or line.startswith("#include"):
                    include_path = yaml.safe_load(line.rstrip().split(" ", 2)[-1])
                    self.merge_with_yaml(include_path)
                else:
                    body = line + file.read()
                    break
        self.merge_with_dict(yaml.safe_load(body) or {})

    def merge_with_dict(self, values):
        for key, value in values.items():
            assert key in self.__dict__, f'unknown config field "{key}"'
            current = getattr(self, key)
            if isinstance(value, dict):
                assert isinstance(current, SimpleConfig)
                current.merge_with_dict(value)
            else:
                setattr(self, key, value)

    def merge_with_dotlist(self, overrides):
        for override in overrides:
            path, text = override.split("=", 1)
            value = yaml.safe_load(text)
            if isinstance(value, str) and value == text.strip():
                try:
                    value = float(text)
                except ValueError:
                    pass
            self.merge_with_dotpath(path, value)

    def merge_with_dotpath(self, path, value):
        keys = path.split(".")
        target = self
        for key in keys[:-1]:
            assert key in target.__dict__, f'unknown config field "{key}"'
            target = getattr(target, key)
            assert isinstance(target, SimpleConfig)
        key = keys[-1]
        assert key in target.__dict__, f'unknown config field "{key}"'
        setattr(target, key, value)

    def check(self):
        for key, value in self.__dict__.items():
            annotation = get_type_hints(type(self))[key]
            if not self.type_matches(value, annotation):
                raise TypeError(
                    f'{type(self).__name__}.{key} expects {annotation}, '
                    f'got {type(value)}'
                )
            if isinstance(value, SimpleConfig):
                value.check()

    @classmethod
    def type_matches(cls, value, annotation):
        if annotation is Any:
            return True
        if annotation is type(None):
            return value is None

        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin in (Union, UnionType):
            return any(cls.type_matches(value, item) for item in arguments)
        if origin in (list, tuple):
            if type(value) is not origin:
                return False
            if not arguments:
                return True
            if origin is tuple and arguments[-1] is not Ellipsis:
                return len(value) == len(arguments) and all(
                    cls.type_matches(item, item_type)
                    for item, item_type in zip(value, arguments)
                )
            return all(cls.type_matches(item, arguments[0]) for item in value)
        if isinstance(annotation, type) and issubclass(annotation, SimpleConfig):
            return isinstance(value, annotation)
        return type(value) is annotation

    def to_dict(self):
        return {
            key: value.to_dict() if isinstance(value, SimpleConfig) else value
            for key, value in self.__dict__.items()
        }

    def to_yaml(self):
        return yaml.dump(
            self.to_dict(), Dumper=NoAliasDumper,
            default_flow_style=False, sort_keys=False)
