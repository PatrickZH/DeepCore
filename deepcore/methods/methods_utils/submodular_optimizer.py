import numpy as np


optimizer_choices = ["NaiveGreedy", "LazyGreedy", "StochasticGreedy", "ApproximateLazyGreedy"]

class optimizer(object):
    def __init__(self, args, index, budget:int, already_selected=[]):
        self.args = args
        self.index = index

        if budget <= 0 or budget > index.__len__():
            raise ValueError("Illegal budget for optimizer.")

        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected


class NaiveGreedy(optimizer):
    def __init__(self, args, index, budget:int, already_selected=[]):
        super(NaiveGreedy, self).__init__(args, index, budget, already_selected)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        for i in range(sum(selected), self.budget):
            if i % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, self.budget))
            greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
            current_selection = greedy_gain.argmax()
            selected[current_selection] = True
            greedy_gain[current_selection] = -np.inf
            if update_state is not None:
                update_state(np.array([current_selection]), selected, **kwargs)
        return self.index[selected]


class LazyGreedy(optimizer):
    def __init__(self, args, index, budget:int, already_selected=[]):
        super(LazyGreedy, self).__init__(args, index, budget, already_selected)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            if i % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, self.budget))
            best_gain = -np.inf
            last_max_element = -1
            while True:
                cur_max_element = greedy_gain.argmax()
                if last_max_element == cur_max_element:
                    # Select cur_max_element into the current subset
                    selected[cur_max_element] = True
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected, **kwargs)
                    break
                new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0]
                greedy_gain[cur_max_element] = new_gain
                if new_gain >= best_gain:
                    best_gain = new_gain
                    last_max_element = cur_max_element
        return self.index[selected]


class StochasticGreedy(optimizer):
    def __init__(self, args, index, budget:int, already_selected=[], epsilon: float=0.9):
        super(StochasticGreedy, self).__init__(args, index, budget, already_selected)
        self.epsilon = epsilon

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        sample_size = max(round(-np.log(self.epsilon) * self.n / self.budget), 1)

        greedy_gain = np.zeros(len(self.index))
        all_idx = np.arange(self.n)
        for i in range(sum(selected), self.budget):
            if i % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, self.budget))

            # Uniformly select a subset from unselected samples with size sample_size
            subset = np.random.choice(all_idx[~selected], replace=False, size=min(sample_size, self.n - i))

            if subset.__len__() == 0:
                break

            greedy_gain[subset] = gain_function(subset, selected, **kwargs)
            current_selection = greedy_gain[subset].argmax()
            selected[subset[current_selection]] = True
            greedy_gain[subset[current_selection]] = -np.inf
            if update_state is not None:
                update_state(np.array([subset[current_selection]]), selected, **kwargs)
        return self.index[selected]


class ApproximateLazyGreedy(optimizer):
    def __init__(self, args, index, budget:int, already_selected=[], beta: float=0.9):
        super(ApproximateLazyGreedy, self).__init__(args, index, budget, already_selected)
        self.beta = beta

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            if i % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, self.budget))
            while True:
                cur_max_element = greedy_gain.argmax()
                max_gain = greedy_gain[cur_max_element]

                new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0]

                if new_gain >= self.beta * max_gain:
                    # Select cur_max_element into the current subset
                    selected[cur_max_element] = True
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected, **kwargs)
                    break
                else:
                    greedy_gain[cur_max_element] = new_gain
        return self.index[selected]




