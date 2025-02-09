import functools
from alive_progress import alive_bar


def with_loading(title: str):
    """
    Decorator to add loading animation to methods.
    
    Args:
        title (str): Title to display during loading
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with alive_bar(title=title, bar=None, stats=False) as bar:
                result = func(self, *args, **kwargs)
                bar()
            return result
        return wrapper
    return decorator