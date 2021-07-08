"""Smith-Waterman graph kernel."""

from operator import itemgetter

import networkx as nx
import numpy as np


class SmithWaterman:
    def __init__(self, normalize: bool = False, gap_cost: float = 1.):
        self.normalize = normalize
        self.gap_cost = gap_cost

    def __compare(self, G_1, G_2, alignment_score):
        """Compute the kernel value (similarity) between two graphs.

        Parameters
        ----------
        G_1 : networkx.Graph
            First graph.
        G_2 : networkx.Graph
            Second graph.

        Returns
        -------
        k : The similarity value between G_1 and G_2.
        """

        if list(nx.simple_cycles(G_1)):
            G1_rxn = [n for n, d in sorted(G_1.in_degree(), key=itemgetter(1))]
        else:
            G1_rxn = list(nx.topological_sort(G_1))

        if list(nx.simple_cycles(G_2)):
            G2_rxn = [n for n, d in sorted(G_2.in_degree(), key=itemgetter(1))]
        else:
            G2_rxn = list(nx.topological_sort(G_2))

        S = np.zeros((len(G1_rxn) + 1, len(G2_rxn) + 1))
        for i in range(1, len(G1_rxn) + 1):
            for j in range(1, len(G2_rxn) + 1):
                match = S[i - 1, j - 1] + (alignment_score if G1_rxn[i - 1] == G2_rxn[j - 1] else 0)
                delete = S[1:i, j].max() - self.gap_cost if i > 1 else 0
                insert = S[i, 1:j].max() - self.gap_cost if j > 1 else 0
                S[i, j] = max(match, delete, insert, 0)
        return S.max()

    def compare(self, G_1, G_2, alignment_score: float = 1.) -> float:
        """Compute the Smith-Waterman value between two graphs.

        A normalized version of the kernel is given by the equation:
        k_norm(G_1, G_2) = k(G_1, G_2) / sqrt(k(G_1,G_1) * k(G_2,G_2))

        Parameters
        ----------
        G_1 : networkx.Graph
            First graph.
        G_2 : networkx.Graph
            Second graph.

        Returns
        -------
        k : The similarity value between G_1 and G_2.
        """

        tmp = self.__compare(G_1=G_1, G_2=G_2, alignment_score=alignment_score)
        if self.normalize:
            tmp = tmp / (np.sqrt(self.__compare(G_1=G_1, G_2=G_1, alignment_score=2) *
                                 self.__compare(G_1=G_2, G_2=G_2, alignment_score=2)))
        return tmp
