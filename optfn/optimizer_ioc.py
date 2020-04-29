import queue
import threading


class OptimizerIOC:
    def __init__(self, optimizer_starter, initial_val_getter):
        self._awaiting_rate = False
        self._optimizer_starter = optimizer_starter
        self._sample_queue = queue.Queue()
        self._reward_queue = queue.Queue()
        self._result = None
        self._is_done = False
        self._update_event = threading.Event()
        self._initial_val_getter = initial_val_getter

        t = threading.Thread(target=self._start)
        t.start()

    @property
    def awaiting_rate(self):
        return self._awaiting_rate

    @property
    def optimizer_starter(self):
        return self._optimizer_starter

    @property
    def result(self):
        return self._result

    @property
    def is_done(self):
        return self._is_done

    def _start(self):
        while True:
            try:
                self._optimizer_starter(self._fitness, self._initial_val_getter())
            except:
                self._is_done = True
                self._update_event.set()
                raise

    def _fitness(self, sample):
        self._sample_queue.put(sample)
        self._update_event.set()
        reward = self._reward_queue.get()
        self._reward_queue.task_done()
        return reward

    def get_sample(self):
        assert not self._awaiting_rate and not self.is_done
        self._awaiting_rate = True
        sample = self._sample_queue.get()
        self._sample_queue.task_done()
        return sample

    def rate_sample(self, fitness):
        assert self._awaiting_rate and not self.is_done
        self._awaiting_rate = False
        self._update_event.clear()
        self._reward_queue.put(fitness)
        self._update_event.wait()