import functools


class FactoryBuilderNotFound(ValueError):
    def __init__(self, v: str, factory: "ObjectFactory" = None) -> None:
        self.v = v
        self.class_name = factory.__class__.__name___

    def __repr__(self) -> str:
        return f"spec {self.v} not registered with {self.class_name}"


class ObjectFactory:
    _registry = {}

    @classmethod
    def register(cls, v: str) -> callable:
        def decorator(fn):
            cls._registry[v.lower()] = fn

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def create(cls, key, **kwargs):
        builder = cls._registry.get(key)
        if not builder:
            raise FactoryBuilderNotFound(key, builder)
        return builder(**kwargs)
