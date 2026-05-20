import copy
import numpy as np
import math
import torch as th


class SimilarityFusion:
    def __init__(self, args):

        self.miRNA_sim_dir = args.miRNA_sim_dir
        self.drug_sim_dir = args.drug_sim_dir
        self.association_m_dir = args.association_m_dir
        self.device = args.device

        self.miRNA = th.from_numpy(np.loadtxt(args.miRNA_sim_dir, dtype=float)).to(args.device)
        self.drug = th.from_numpy(np.loadtxt(args.drug_sim_dir, dtype=float)).to(args.device)
        self.A = th.from_numpy(np.loadtxt(args.association_m_dir, dtype=int)).to(args.device)

    @staticmethod
    def GIP_sm(A):


        w1 = np.linalg.norm(A, axis=1)
        width_sum = np.sum(np.square(w1))
        Y_sm = A.shape[0] / width_sum
        G = np.zeros((A.shape[0], A.shape[0]))
        for i in range(G.shape[0]):
            for j in range(i, G.shape[1]):
                G[i, j] = math.exp((-Y_sm) * np.square(np.linalg.norm(A[i] - A[j])))
                G[j, i] = G[i, j]
        return G

    @staticmethod
    def GIP_m(A):

        w1 = np.linalg.norm(A, axis=0)
        width_sum = np.sum(np.square(w1))
        Y_m = A.shape[1] / width_sum
        G = np.zeros((A.shape[1], A.shape[1]))
        for i in range(G.shape[0]):
            for j in range(i, G.shape[1]):
                G[i, j] = math.exp((-Y_m) * np.square(np.linalg.norm(A[:, i] - A[:, j])))
                G[j, i] = G[i, j]
        return G

    @staticmethod
    def InSm(sm1, sm2, w):

        return w * sm1 + (1 - w) * sm2

    def calculate_fusion(self, w):

        B = self.GIP_sm(self.A.cpu().numpy())
        C = self.GIP_m(self.A.cpu().numpy())


        miRNA_fused = self.miRNA.cpu().numpy()
        drug_fused = self.drug.cpu().numpy()


        return th.from_numpy(miRNA_fused).to(self.device), th.from_numpy(drug_fused).to(self.device)
