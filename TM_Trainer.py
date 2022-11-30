import numpy as np
import matplotlib.pyplot as plt


# total number of model parameters is
# s^2 len(Symbol)^2 len(Dir) - s len(Symbol)

# dirs: left, right, stay-put

def inverse_broadcast(f, arr1, arr2):
    return np.transpose(
        f(  np.transpose(arr1),
            np.transpose(arr2)
        )
    )

class TM:
    def __init__(self, s, alphabet_size):
        self.s = s
        self.alphabet_size = alphabet_size
        self.trans_table = np.random.random([s, alphabet_size, s, alphabet_size, 3])
        # normalization
        denom = np.sum(self.trans_table, axis=(2,3,4))
        self.trans_table = inverse_broadcast(np.divide, self.trans_table, denom)

    def throw_die(self, probs):
        die = np.random.choice(range(self.s * self.alphabet_size * 3), p=probs.flatten())
        dir = die%3
        new_sym = (die//3)%self.alphabet_size
        next_state = (die//(3*self.alphabet_size))%self.s
        return next_state, new_sym, dir

    def simulate(self, w, time_budget, space_budget):
        self.cur_state = 0
        self.head = 0
        self.tape = np.zeros(space_budget, dtype=np.int8)
        self.tape[:len(w)] = w
        
        trans_stats = np.zeros(shape=[self.s, self.alphabet_size, self.s, self.alphabet_size, 3])
        time_tics = 1
        while self.s - self.cur_state > 2 and time_budget >= time_tics and self.head < space_budget:
            # print repr
            config_str = np.array2string(self.tape[:self.head], separator='') \
            + chr(ord('A')+self.cur_state) \
            + np.array2string(self.tape[self.head:], separator='')
            # print(config_str)
            # choose a transition
            cur_sym = self.tape[self.head]
            probs = self.trans_table[self.cur_state, cur_sym]
            next_state, new_sym, dir = self.throw_die(probs)
            # keep track of transitions
            trans_stats[self.cur_state, cur_sym, next_state, new_sym, dir] += 1
            # update
            self.cur_state = next_state
            self.tape[self.head] = new_sym
            # do not move left if you are already at the leftmost position
            if self.head != 0 or dir != 0:
                self.head += dir - 1
            time_tics += 1

        if self.cur_state == self.s - 1:
            return 'accept', trans_stats
        elif self.cur_state == self.s - 2:
            return 'reject', trans_stats
        else:
            return 'no answer', trans_stats


# TODO: dynamically adjust budgets so that
# accept and reject probs sum up close to 1
class TM_Trainer():
    
    def __init__(self, s, alph_size, learning_rate, converge_threshold, sample_size):
        self.learning_rate = learning_rate
        self.converge_threshold = converge_threshold
        self.sample_size = sample_size
        self.model_params = np.random.rand(s, alph_size, s, alph_size, 3)*1000 + 1000
        self.tm = TM(s, alph_size)
        self.time_budget = 1024
        self.space_budget = self.time_budget
    
    # data is given as a python dict
    def input_data(self, data):
        self.data = data

    def update_probs(self):
        denom = np.sum(self.model_params, axis=(2,3,4))
        self.tm.trans_table = inverse_broadcast(np.divide, self.model_params, denom)

    # prob_type: 0 for prob. of reject, 1 for prob. of accept
    # label:     0 for reject, 1 for accept
    # grad:      gradient of prob. of {prob_type} with respect to model params
    # loss:      (Pr(x)_ac - label)^2 + (Pr(x)_rej - (1-label))^2
    # we calculate only one term of the loss for the sake of speed
    def train(self):
        total_loss_grad = np.zeros(shape=np.shape(self.model_params))
        total_loss = 0
        grad_norm = 1
        graph = []
        #while grad_norm > self.converge_threshold:
        for _ in range(40):
            total_loss_grad.fill(0)
            total_loss = 0
            # iterate over all data
            for w, label in self.data.items():
                grad, prob, prob_type = self.single_data(list(w))
                # print('Prob of', prob_type, 'is', prob)
                # total_loss += (prob - label)**2 + (prob - (1-label))**2
                if prob_type == 'accept':
                    total_loss_grad += (prob - label) * grad
                    total_loss += (prob - label)**2
                else:
                    total_loss_grad += (prob - (1 - label)) * grad
                    total_loss += (prob - (1-label))**2
            # gradient descent
            self.model_params -= self.learning_rate * total_loss_grad
            self.update_probs()
            graph.append(total_loss)
            print('Total loss:', total_loss)
            grad_norm = np.linalg.norm(total_loss_grad.flatten())
            #print('Grad norm: ', np.linalg.norm(total_loss_grad))
            #print('model_params[0][0]:', self.model_params[0][0])
        return graph

    
    def single_data(self, w):
        counters = {
            'accept': 0,
            'reject': 0,
            'total': 0
        }
        results = []
        while counters['accept'] < self.sample_size and counters['reject'] < self.sample_size:
            res = self.tm.simulate(w, self.time_budget, self.space_budget)
            counters['total'] += 1
            counters[res[0]] += 1
            results.append(res)
        
        prob_type = 'accept'
        prob = None
        if counters['accept'] > counters['reject']:
            results = [x for x in results if x[0] == 'accept']
            prob = counters['accept'] / counters['total']
        else:
            prob_type = 'reject'
            results = [x for x in results if x[0] == 'reject']
            prob = counters['reject'] / counters['total']

        trans_counts = [res[1] for res in results]
        avg_trans_counts = np.average(np.array(trans_counts), axis=0)
        ratios = avg_trans_counts / self.model_params
        sum_ratio = np.sum(avg_trans_counts, axis=(2,3,4)) / np.sum(self.model_params, axis=(2,3,4))
        grad = inverse_broadcast(np.subtract, ratios, sum_ratio) * prob
        return grad, prob, prob_type



# tm = TM(10,2)
# tm.simulate([1,1,0,1], 10, 10)

input = {
    '1': 1,
    '20': 1,
    '10': 1,
    '01': 0,
    '011': 0,
    '210': 1,
    '211': 1,
    '022': 0,
    '0': 0
}

trainer = TM_Trainer(3, 3, learning_rate=1e6, converge_threshold=2e-5, sample_size=100)
trainer.input_data(input)
graph = trainer.train()
plt.plot(graph)
plt.show()

# FIXME: learning happens with single data point, however, model params quickly go to negative realm.
# Rescaling of params seem to hurt the convergence, we should find another solution
# Idea: map the square of params to probs

# convergence slows down with increasing number of states, substantially

# FIXME: learning does not happen for problems which requires to look at the second symbol of the input