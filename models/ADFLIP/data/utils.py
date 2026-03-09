import bisect
import time
from typing import Callable, Any
from sortedcontainers import SortedList


class TimeSortedCache:
    """
    Cache that sorts elements by the time it took to process them.
    If it is out of space, it removes the cheapest element.
    """

    def __init__(self, capacity: int, parse_fn: Callable[[Any], Any]):
        self.capacity = capacity
        self.parse_fn = parse_fn
        self.times = []  # List to store processing times
        self.cache = {}  # Dictionary to store key-value pairs

    def get(self, key: Any) -> Any:
        if key not in self.cache:
            return self._process_and_cache(key)

        return self.cache[key][1]

    def _process_and_cache(self, key: Any) -> Any:
        start_time = time.perf_counter()

        value = self.parse_fn(key)

        process_time = time.perf_counter() - start_time

        if self.capacity > -1 and (len(self.cache) >= self.capacity):
            if self.times[0] < process_time:
                # Remove the cheapest item
                cheapest_time = self.times.pop(0)
                cheapest_key = next(
                    k for k, v in self.cache.items() if v[0] == cheapest_time
                )
                del self.cache[cheapest_key]
                self._insert_item(key, value, process_time)

        else:
            self._insert_item(key, value, process_time)

        return value

    def __len__(self):
        return len(self.cache)

    def _insert_item(self, key, value, process_time):
        raise NotImplementedError

    def __str__(self):
        return f"TimeSortedCache(capacity={self.capacity}, items={len(self.cache)})"


class TimeSortedCacheBisect(TimeSortedCache):
    def _insert_item(self, key, value, process_time):
        bisect.insort(self.times, process_time)
        self.cache[key] = (process_time, value)


class TimeSortedCacheRBT(TimeSortedCache):
    def __init__(self, capacity: int, parse_fn: Callable[[Any], Any]):
        super().__init__(capacity, parse_fn)
        self.times = SortedList()

    def _insert_item(self, key, value, process_time):
        self.times.add(process_time)
        self.cache[key] = (process_time, value)
