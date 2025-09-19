import time
import functools
import logging
from typing import Callable, Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager and decorator for timing code execution.
    """
    
    def __init__(self, name: str = "operation", log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.log(self.log_level, f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        logger.log(self.log_level, f"Completed {self.name} in {self.elapsed:.3f}s")
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(f"{func.__name__}", self.log_level):
                return func(*args, **kwargs)
        return wrapper


def timeit(func: Callable = None, *, name: str = None, log_level: int = logging.INFO) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time (when used without parentheses)
        name: Custom name for the timer
        log_level: Logging level for timing messages
        
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        timer_name = name or f.__name__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with Timer(timer_name, log_level):
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        # Called with arguments: @timeit(name="custom")
        return decorator
    else:
        # Called without arguments: @timeit
        return decorator(func)


class ProfileTimer:
    """
    Accumulating timer for profiling repeated operations.
    """
    
    def __init__(self):
        self.timers: Dict[str, list] = {}
        self.current_starts: Dict[str, float] = {}
    
    def start(self, name: str) -> None:
        """Start timing an operation."""
        self.current_starts[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing an operation and record the elapsed time."""
        if name not in self.current_starts:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.perf_counter() - self.current_starts[name]
        
        if name not in self.timers:
            self.timers[name] = []
        
        self.timers[name].append(elapsed)
        del self.current_starts[name]
        
        return elapsed
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named timer."""
        if name not in self.timers:
            return {}
        
        times = self.timers[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timers."""
        return {name: self.get_stats(name) for name in self.timers.keys()}
    
    def reset(self, name: str = None) -> None:
        """Reset timers (specific timer or all if name is None)."""
        if name is None:
            self.timers.clear()
            self.current_starts.clear()
        else:
            self.timers.pop(name, None)
            self.current_starts.pop(name, None)
    
    def context(self, name: str):
        """Context manager for timing operations."""
        return ProfileTimerContext(self, name)


class ProfileTimerContext:
    """Context manager for ProfileTimer."""
    
    def __init__(self, profile_timer: ProfileTimer, name: str):
        self.profile_timer = profile_timer
        self.name = name
    
    def __enter__(self):
        self.profile_timer.start(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profile_timer.stop(self.name)


def measure_performance(func: Callable, *args, iterations: int = 1, **kwargs) -> Dict[str, Any]:
    """
    Measure performance of a function over multiple iterations.
    
    Args:
        func: Function to measure
        *args: Arguments for the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments for the function
        
    Returns:
        Dictionary with performance statistics
    """
    times = []
    results = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in iteration {i}: {e}")
            results.append(None)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'iterations': iterations,
        'times': times,
        'total_time': sum(times),
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'results': results,
        'success_rate': sum(1 for r in results if r is not None) / len(results)
    }


# Global profile timer instance
profile_timer = ProfileTimer()