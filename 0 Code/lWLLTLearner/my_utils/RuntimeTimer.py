from typing import Dict, Tuple, List, Any
import time
from typing_extensions import runtime

class RuntimeTimer():

    def __init__(self, name: str="main"):
        self.name = name
        self.runtimes_dict:     Dict[str, float] = {}
        self._last_start_time:   Dict[str, float] = {}

    def start_timer(self, name: str) -> float:
        start_time = time.time()        
        self._last_start_time[name] = start_time

        return start_time

    def stop_timer(self, name: str, append: bool = False) -> Tuple[float, float, float]:
        """ Stop a running timer. If 'append', the time will be accumulated.
        Returns the current time, runtime since the last call, and the accumulated runtime.
        If not 'append' the last two values are equal. """
        stop_time = time.time()
        run_time  = stop_time - self._last_start_time[name]
        return_tuple = tuple()

        if append and self.runtimes_dict.get(name) is not None:
            self.runtimes_dict[name] += run_time    
            return_tuple = stop_time, runtime, self.runtimes_dict[name]
        else:
            self.runtimes_dict[name] = run_time
            return_tuple = stop_time, runtime, runtime
        
        return return_tuple

    def get_time(self, name: str) -> str:
        
        saved_time = self.runtimes_dict.get(name)
        if saved_time is None:
            return f"No timer named {name} was set!"
        else:
            return convert_s_to_h_m_s(saved_time)

    def get_name_runtime_str(self, delimiter:str = "\t") -> str:
        """ Returns a string like 'name0\truntime0{delimiter}name1...' """
        return_string: str = [f"{name}\t{convert_s_to_h_m_s(value)}{delimiter}" for name, value in self.runtimes_dict.items()]
        # Remove the last delimiter.
        return_string = return_string[:-len(delimiter)]

        return return_string

    def get_all_runtime_names(self) -> List[str]:
        """ Returns a List[str] of names. """
        return list(self.runtimes_dict.keys())

    def get_all_runtime_values(self, convert_float_to_str: bool = True) -> List[Any]:
        """ Returns a List[float] of all runtimes. """
        if convert_float_to_str:
            return [convert_s_to_h_m_s(r) for r in self.runtimes_dict.values()]
        else:
            return list(self.runtimes_dict.values())

def convert_s_to_h_m_s(sec: float) -> str:
        """
        Returns a string interpretation of the given seconds in
        non-negative hours, minutes, seconds and milisconds.

        If hours > 0, do not show seconds or miliseconds.
        If hours = 0 but min > 0, do not show hours or miliseconds.
        If only seconds and miliseconds are displayed, round all miliseconds smaller than 'threshold' to zero.
        """
        sec = sec % (24 * 3600)
        hours = sec // 3600
        sec %= 3600
        minutes = sec // 60
        sec %= 60
        milli_sec = (sec % 1) * 1000
        only_sec = sec // 1

        threshold: int = 3

        if hours != 0.0:
            return "%2d h %2d min" % (hours, minutes)

        if minutes != 0.0:
            return "%2d min %2d sec" % (minutes, only_sec)

        if only_sec == 0.0:
            if round(milli_sec, threshold) == 0.0:
                return "< 1 ms"
            return "%3d ms" % milli_sec
        elif round(milli_sec, threshold) == 0.0:
            return "%2d sec" % only_sec

        return "%2d sec %3d ms" % (only_sec, milli_sec)

import enum
class RT(enum.Enum):
   Total = "RT from start to finish."
   Method = "RT of the actual computations."
   Zero = "RT of executing no code."

def _demo():
    test_timer = RuntimeTimer()

    test_timer.start_timer(RT.Total)

    time.sleep(3)

    test_timer.start_timer(RT.Method)
    prod = 1
    for i in range(1, 100000):
        prod *= i
    test_timer.stop_timer(RT.Method)

    time.sleep(3)
    test_timer.start_timer(RT.Zero)
    test_timer.stop_timer(RT.Zero)
    test_timer.stop_timer(RT.Total)

    print(test_timer.get_name_runtime_str(delimiter='\n'))

if __name__=="__main__":
    _demo()