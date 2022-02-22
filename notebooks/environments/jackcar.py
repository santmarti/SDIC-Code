import numpy as np
import sys
import math

from gym.envs.toy_text import discrete

def clipped_poisson(lam, max_k):
    """
    Return poisson PMF clipped at max_k with remaining tail probability
    placed at max_k.
    """
    pmf = np.zeros(max_k + 1)
    for k in range(max_k):
        pmf[k] = math.exp(-lam) * lam**k / math.factorial(k)
    pmf[max_k] = 1 - np.sum(pmf)
    
    return pmf    


class JackCarRentalEnv(discrete.DiscreteEnv):
    """ Example 4.2 from Reinforcement Learning: An Introduction by Sutton and Barto.
    """

    def build_pmfs(self, lambda_request, lambda_return, max_cars):
        """
        Return p(new_rentals, returns | initial_cars) as numpy array:
            p[initial_cars, new_rentals, returns]
        """
        pmf = np.zeros((max_cars+1, max_cars+1, max_cars+1))

        for init_cars in range(max_cars + 1):
            new_rentals_pmf = clipped_poisson(lambda_request, init_cars)
            for new_rentals in range(init_cars + 1):
                max_returns = max_cars - init_cars + new_rentals
                returns_pmf = clipped_poisson(lambda_return, max_returns)
                for returns in range(max_returns + 1):
                    p = returns_pmf[returns] * new_rentals_pmf[new_rentals]
                    pmf[init_cars, new_rentals, returns] = p
                    
        return pmf

    def get_transition_model(self, s, a):
        """
        Inputs: state as 2-tuple / action as -2,-1,0,1,2 [-max,max]
        Returns a 2-tuple:
            1. p(s'| s, a) as dictionary:
                keys = s'
                values = p(s' | s, a)
            2. E(r | s, a, s') as dictionary:
                keys = s'
                values = E(r | s, a, s')
        """
        s = (s[0] - a, s[1] + a)           # move a cars from loc1 to loc2        
        s = np.clip(s,0,self.max_cars)     # impossible actions are cliped to possible states

        move_reward = -math.fabs(a) * 2  # ($2) per car moved
        t_prob, expected_r = ([{}, {}], [{}, {}])
        for loc in range(2):
            morning_cars = s[loc]
            rent_return_pmf = self.rent_return_pmf[loc]
            for rents in range(morning_cars + 1):
                max_returns = self.max_cars - morning_cars + rents
                for returns in range(max_returns + 1):
                    p = rent_return_pmf[morning_cars, rents, returns]
                    if p < 1e-5:
                        continue
                    s_prime = morning_cars - rents + returns
                    r = rents * 10
                    t_prob[loc][s_prime] = t_prob[loc].get(s_prime, 0) + p
                    expected_r[loc][s_prime] = expected_r[loc].get(s_prime, 0) + p * r
        
        # join probabilities and expectations from loc1 and loc2
        t_model, r_model = ({}, {})
        for s_prime1 in t_prob[0]:
            for s_prime2 in t_prob[1]:
                p1 = t_prob[0][s_prime1]  # p(s' | s, a) for loc1
                p2 = t_prob[1][s_prime2]  # p(s' | s, a) for loc2
                t_model[(s_prime1, s_prime2)] = p1 * p2
                # expectation of reward calculated using p(s', r | s, a)
                # need to normalize by p(s' | s, a)
                norm_E1 = expected_r[0][s_prime1] / p1
                norm_E2 = expected_r[1][s_prime2] / p2

                norm_E1 = 10*round(norm_E1 / 10)
                norm_E2 = 10*round(norm_E2 / 10)

                r_model[(s_prime1, s_prime2)] = norm_E1 + norm_E2 + move_reward
                
        return t_model, r_model





    def __init__(self, max_cars = 4, rents_per_day = (3,4), returns_per_day = (3,2) ):
        """ The environment is a DiscreteEnv Gym with the following members:
        - nS: number of states
        - nA: number of actions
        - P: transitions (*)
        - isd: initial state distribution (**)
        (*) dictionary dict of dicts of lists, where
            P[s][a] == [(probability, nextstate, reward, done), ...]
        (**) list or array of length nS
        """

        self.max_cars = max_cars
        self.max_move_cars = int(max_cars / 4)
        self.grid_shape = (max_cars+1,max_cars+1)
        self.rents_per_day = rents_per_day
        self.returns_per_day = returns_per_day

        print("Initialized JackCarRental Environment : %d max_cars %d max_moving cars"%(max_cars,self.max_move_cars))

        nS = np.prod(self.grid_shape)
        nA = len(np.arange(-self.max_move_cars, self.max_move_cars + 1))

        # pre-build the rentals/returns pmf for each location
        self.rent_return_pmf = [self.build_pmfs(self.rents_per_day[i], self.returns_per_day[i], max_cars) for i in [0,1] ]

        P = {}
        for s_index in range(nS):
            s = np.unravel_index(s_index, self.grid_shape)
            P[s_index] = { a : [] for a in range(nA) }

            max_a = min(self.max_move_cars, s[0], max_cars-s[1])
            min_a = max(-self.max_move_cars, -s[1], -(max_cars-s[0]))
    
            for a_real in range(min_a, max_a+1):
                a = a_real + self.max_move_cars
                state_real = np.array(s) + np.array([-a_real, a_real])

                t_model, r_model = self.get_transition_model(s, a_real)
                for sp in t_model:
                    p = t_model[sp]
                    r = r_model[sp]
                    sp_index = np.ravel_multi_index(sp,self.grid_shape)
                    P[s_index][a].append([p, sp_index, r, False])

        isd = np.zeros(nS)
        isd[int(nS/2)] = 1.0
        super(JackCarRentalEnv, self).__init__(nS, nA, P, isd)

    def render(self):
        print("nothing to render")


