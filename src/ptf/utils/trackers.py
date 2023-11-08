from .typing_hints import MetricStateDict, MetricCache

from collections import deque


class _BaseTracker:
    r"""
    A Tracker can record some certain metirc values during the training proce-
    dure, `_BaseTracker` is the base class for all tracker classes. 
    
    The lastest value and according step are recorded by :attr cache: and :attr step:.
    The smoothed value over the training procedure can be obtained by :method get_smoothed_value:. 
    """
    def __init__(self) -> None:
        self.cache: MetricCache = None
        self.step: int = 0
    
    def add(self, x: float) -> None:
        raise NotImplementedError
    
    def report(self) -> float:
        return self.get_smoothed_value()
    
    def get_smoothed_value(self) -> float:
        r"""
        Any steam-based tracker does not record all history data, but update the
        smoothed value in a recursive way. This method return the smoothed value
        computed by some specific implementaion.
        """
        raise NotImplementedError
    
    def state_dict(self) -> MetricStateDict:
        return {
            "cache": self.cache,
            "step": self.step
        }
    
    def load_state_dict(self, state_dict: MetricStateDict) -> None:
        self.cache = state_dict["cache"]
        self.step = state_dict["step"]


class SimpleTracker(_BaseTracker):
    r"""
    Simple Tracker just track the latest value from the data stream. This type of
    tracker is suitable for situation in which we focus on *EVERY* data, such as 
    tracking evaluating results like classification correctness, fid score, etc.
    """
    def __init__(self) -> None:
        super().__init__()
        self.cache: float = 0.0
    
    def add(self, x):
        self.cache = x
        self.step += 1
    
    def get_smoothed_value(self) -> float:
        return self.cache


class HistoryTracker(_BaseTracker):
    r"""
    History Tracker will store all history data into cache. The smoothed value is
    the latest value it received.
    """
    def __init__(self) -> None:
        super().__init__()
        cache: list[float] = []
    
    def add(self, x):
        self.cache.append(x)
        self.step += 1
    
    def get_smoothed_value(self) -> float:
        return self.cache[-1]


class FiniteHistoryTracker(HistoryTracker):
    r"""
    Finite History Tracker stores history data into cache, but only keeps finite 
    number of histories. The smoothed value is the mean over the finite window.
    """   
    
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.cache = deque([])
        self.capacity = capacity
    
    def add(self, x):
        super().add(x)
        if len(self.cache) > self.capacity:
            self.cache.popleft()
    
    def get_smoothed_value(self) -> float:
        return sum(self.cache) / len(self.cache)


class AverageTracker(_BaseTracker):
    r"""
    Average Tracker will tracking the average value over all data stream. The 
    recursive update formulate is:
        avg_t = [(t - 1) * avg_{t-1} + x_t] / t
    """
    def __init__(self) -> None:
        super().__init__()
        self.cache: float = 0.0
    
    def add(self, x):
        self.cache = self.step * self.cache + x
        self.step += 1
        self.cache = self.cache / self.step 
    
    def get_smoothed_value(self) -> float:
        return self.cache


class PeriodAverageTracker(_BaseTracker):
    r"""
    Period Average Tracker reports period mean, it empty the cache list every 
    several steps.
    """
    def __init__(self, period: int) -> None:
        super().__init__()
        self.cache: list[float] = []
        self.period = period 
    
    def add(self, x):
        if len(self.cache) == self.period:
            self.cache = []
        self.cache.append(x)
    
    def get_smoothed_value(self) -> float:
        return sum(self.cache) / len(self.cache)


class IncrementalAverageTracker(_BaseTracker):
    r"""
    Incremental Average Tracker reports the mean of recorded history, every time 
    it is called, it also empty its cache space.
    """
    def __init__(self) -> None:
        super().__init__()
        self.cache: list[float] = []
    
    def report(self) -> float:
        avg = super().report()
        self.cache = []
        return avg
    
    def get_smoothed_value(self) -> float:
        return sum(self.cache) / len(self.cache)
    
    def add(self, x):
        self.cache.append(x)
        self.step += 1


#* Alias

class LossTracker(IncrementalAverageTracker): ...
class MetricTracker(SimpleTracker): ...
