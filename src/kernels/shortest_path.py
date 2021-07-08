"""Shortest-Path graph kernel."""

import networkx as nx
import numpy as np


class ShortestPath:
    def __init__(self, normalize: bool = False):
        self.normalize = normalize

    def __compare(self, G_1, G_2):
        """Compute the kernel value (similarity) between two graphs.

        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.

        Returns
        -------
        k : The similarity value between g1 and g2.
        """
        # Diagonal superior matrix of the floyd warshall shortest
        # paths:
        fwm1 = np.array(nx.floyd_warshall_numpy(G_1))
        fwm1 = np.where(fwm1 == np.inf, 0, fwm1)
        fwm1 = np.where(fwm1 == np.nan, 0, fwm1)
        fwm1 = np.triu(fwm1, k=1)
        bc1 = np.bincount(fwm1.reshape(-1).astype(int))

        fwm2 = np.array(nx.floyd_warshall_numpy(G_2))
        fwm2 = np.where(fwm2 == np.inf, 0, fwm2)
        fwm2 = np.where(fwm2 == np.nan, 0, fwm2)
        fwm2 = np.triu(fwm2, k=1)
        bc2 = np.bincount(fwm2.reshape(-1).astype(int))

        # Copy into arrays with the same length the non-zero shortests
        # paths:
        v1 = np.zeros(max(len(bc1), len(bc2)) - 1)
        v1[range(0, len(bc1) - 1)] = bc1[1:]

        v2 = np.zeros(max(len(bc1), len(bc2)) - 1)
        v2[range(0, len(bc2) - 1)] = bc2[1:]
        return np.sum(v1 * v2)

    def compare(self, G_1, G_2) -> float:
        """Compute the normalized kernel value between two graphs.

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

        tmp = self.__compare(G_1=G_1, G_2=G_2)
        if self.normalize:
            tmp = tmp / (np.sqrt(self.__compare(G_1=G_1, G_2=G_1) *
                                 self.__compare(G_1=G_2, G_2=G_2)))
        return tmp
