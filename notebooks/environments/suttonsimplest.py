import numpy as np
import sys
import math

from gym.envs.toy_text import discrete


class SuttonSimplestEnv(discrete.DiscreteEnv):
    """ Interactive Demo in the Sutton talk: https://www.youtube.com/watch?v=ggqnxyjaKe4&t=3736s
    """

    def __init__(self):
        """ The environment is a DiscreteEnv 
        """
        nS,nA = 2,2
        A,B = 0,1

        P = {}
        P[A] = { a : [] for a in range(nA) }
        P[B] = { a : [] for a in range(nA) }

        P[A][0].append([1, A, 10, False])
        
        P[A][1].append([0.2, A, -10, False])
        P[A][1].append([0.8, B, -10, False])

        P[B][0].append([0.8, A, 40, False])
        P[B][0].append([0.2, B, 40, False])

        P[B][1].append([0.2, B, 20, False])
        P[B][1].append([0.8, A, 20, False])

        self.shape = (1)

        isd = np.zeros(nS)
        isd[A] = 1.0
        super(SuttonSimplestEnv, self).__init__(nS, nA, P, isd)

    def render(self):
        print("nothing to render")


