import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class DependencyInjectionError(RuntimeError):
    pass


class DependencyInjectionContainer:
    def __init__(self) -> None:
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def _key(self, key: Any) -> str:
        if isinstance(key, str):
            return key
        if isinstance(key, type):
            return key.__qualname__
        return str(key)

    def register_singleton(self, key: Any, instance: Any) -> None:
        self._singletons[self._key(key)] = instance

    def register_factory(self, key: Any, factory: Callable[[], Any]) -> None:
        self._factories[self._key(key)] = factory

    def resolve(self, key: Any) -> Any:
        key_name = self._key(key)
        if key_name in self._singletons:
            return self._singletons[key_name]
        if key_name in self._factories:
            return self._factories[key_name]()
        raise DependencyInjectionError(f"Dependency not found: {key_name}")

    def build(self, target: Type[T]) -> T:
        constructor = inspect.signature(target.__init__)
        kwargs: Dict[str, Any] = {}
        for name, parameter in constructor.parameters.items():
            if name == "self":
                continue
            annotation = parameter.annotation
            if annotation is inspect._empty:
                if name in self._singletons:
                    kwargs[name] = self._singletons[name]
                    continue
                raise DependencyInjectionError(
                    f"Cannot resolve constructor parameter '{name}' for {target.__name__}."
                )
            try:
                kwargs[name] = self.resolve(annotation)
            except DependencyInjectionError:
                if name in self._singletons:
                    kwargs[name] = self._singletons[name]
                else:
                    raise
        return target(**kwargs)
