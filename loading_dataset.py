import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import KFold
from configs import get_args
import torch.nn as nn
from GIP import SimilarityFusion
class Data_Process(nn.Module):

    def __init__(self, args):
        super(Data_Process, self).__init__()
        self.args = args
        self.m_linear = nn.Linear(args.m_d, args.d_h, bias=True)
        self.d_linear = nn.Linear(args.d_d, args.d_h, bias=True)
        self.fusion = SimilarityFusion(args)


    def Embedding(self, m_sim, d_sim):


        m_embed = self.m_linear(m_sim)
        d_embed = self.d_linear(d_sim)

       
        return m_embed, d_embed

    def get_all_the_samplest(self, association_m, neg_ratio):


        positive_samples = (association_m != 0).nonzero(as_tuple=False)
        negative_samples = (association_m == 0).nonzero(as_tuple=False)


        positive_labels = th.ones(positive_samples.size(0), 1, device=self.args.device,dtype=th.long)
        negative_labels = th.zeros(negative_samples.size(0), 1, device=self.args.device, dtype=th.long)


        num_positive = positive_samples.size(0)
        num_negative_to_select = num_positive * neg_ratio
        selected_indices = th.randint(0, negative_samples.size(0), (num_negative_to_select,), device=self.args.device)
        selected_negative_samples = negative_samples[selected_indices]


        final_samples = th.cat([positive_samples, selected_negative_samples], dim=0)
        final_labels = th.cat([positive_labels, negative_labels[selected_indices]], dim=0)


        return final_samples, final_labels

    def forward(self):




        m_sim, d_sim = self.fusion.calculate_fusion(self.args.G_weight)
        association_m = th.from_numpy(np.loadtxt(self.args.association_m_dir, dtype=int)).to(self.args.device)

        m_sim = m_sim.float()
        d_sim = d_sim.float()


        final_ids, final_labels = self.get_all_the_samplest(association_m, self.args.neg_ratio)

        #
        m_embed, d_embed = self.Embedding(m_sim, d_sim)

        return final_ids, final_labels, m_embed, d_embed


class Data_divide(nn.Module):

    def __init__(self, args, final_ids, final_labels):
        super(Data_divide, self).__init__()
        self.args = args


        self.final_ids = final_ids
        self.final_labels = final_labels

    def forward(self):

        train_indices = []
        test_indices = []

        train_ids_5 = []
        test_ids_5 = []
        train_labels_5 = []
        test_labels_5 = []

        num_samples, _ = self.final_ids.shape

        kf = KFold(n_splits=self.args.fold, shuffle=True, random_state=1)


        for train_index, test_index in kf.split(np.arange(num_samples)):

            train_indices.append(th.tensor(train_index, device=self.args.device))
            test_indices.append(th.tensor(test_index, device=self.args.device))

        for fold_idx in range(self.args.fold):

            train_idx = train_indices[fold_idx].to(self.args.device)
            test_idx = test_indices[fold_idx].to(self.args.device)


            train_ids = self.final_ids[train_idx]
            test_ids = self.final_ids[test_idx]

            train_labels = self.final_labels[train_idx]
            test_labels = self.final_labels[test_idx]

            train_ids_5.append(train_ids)
            test_ids_5.append(test_ids)
            train_labels_5.append(train_labels)
            test_labels_5.append(test_labels)

        return train_ids_5, test_ids_5, train_labels_5, test_labels_5



