from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch.nn.functional as F

import math
import torch
from torch import nn
import numpy as np

class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    if queries.shape[1]>4: #time intervals exist
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start=-int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx


                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries)+self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size]
                        q = self.get_queries(these_queries)
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)
                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        
                        scores = q @ rhs 
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention1, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class TeLM(TKBCModel):
    """2nd-grade Temporal Knowledge Graph Embeddings using Geometric Algebra
        :::     Scoring function: <h, r, t_conjugate, T>
        :::     2-grade multivector = scalar + Imaginary * e_1 + Imaginary * e_2 + Imaginary * e_3 + Imaginary * e_12
    """
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(TeLM, self).__init__()
        
        self.sa1 = SpatialAttention()
        
        self.sizes = sizes
        self.rank = rank
        self.W = nn.Embedding(rank,1,sparse=True)
        self.W.weight.data *= 0
        self.X = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (800, 800, 800)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[0], sizes[0], sizes[0], sizes[3]] # without no_time_emb
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size
        self.embeddings[5].weight.data *= init_size
        self.embeddings[6].weight.data *= init_size
        
        self.conv21 = torch.nn.Conv2d(4, 2, kernel_size=1, stride=1, bias=True)
        self.conv22 = torch.nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=True)

        self.sum_pool1 = torch.nn.LPPool2d(1,(4,1),stride=1)
        
        self.sum_pool = torch.nn.LPPool1d(1,70,stride=70)
        
        self.linear_E = torch.nn.Linear(2400,56000)
        self.linear_R = torch.nn.Linear(2400,56000)
        self.linear_d = torch.nn.Linear(2400,56000)
        
        self.bn_MRN_e1 = torch.nn.BatchNorm2d(2)
        self.bn_DRN_e1 = torch.nn.BatchNorm2d(4)
        
        self.branch_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
        self.branch_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
        self.branch_d = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU())
            
        self.branch1_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_r = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_d = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_s = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch1_s1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
            
        self.branch2_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch2_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
            
        self.branch3_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch3_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())   
            
        self.branch4_e1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_r = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_d = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        self.branch4_s = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1, bias=True),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU())
        
        self.bn1 = torch.nn.BatchNorm1d(800)
        
        self.w = nn.Parameter(torch.ones(2))
        
        self.pre_train = pre_train
	
        if self.pre_train:
            self.embeddings[0].weight.data[:,self.rank:self.rank*3] *= 0
      #      self.embeddings[1].weight.data[:,self.rank:self.rank*3] *= 0
      #      self.embeddings[2].weight.data[:,self.rank:self.rank*3] *= 0
        

        self.no_time_emb = no_time_emb

        self.time_granularity = time_granularity

    @staticmethod
    def has_time():
        return True


    def score(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rnt = self.embeddings[6](x[:, 3] // self.time_granularity)
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity)
        #rnt = self.embeddings[3](x[:, 1])
        
        d = time.view(-1,1,20,40) 
#MRN_e1        
        lhs1 = self.embeddings[3](x[:, 0])
        lhs2 = self.embeddings[4](x[:, 0])
        lhs3 = self.embeddings[5](x[:, 0])
       
        h1 = lhs.view(-1,1,1,800)
        h2 = lhs1.view(-1,1,1,800)
        h3 = lhs2.view(-1,1,1,800)
        h4 = lhs3.view(-1,1,1,800)
       
        h = torch.cat((h1,h2,h3,h4),1)
        #print(h.size())
        #e1 = e1.view(-1, 3, 1, 200)
        h = self.conv21(h)
        h = self.bn_MRN_e1(h)
        h = F.relu(h)
        h = self.conv22(h)  
        e1 = h.view(-1,1,20,40)     
      
        r = rel.view(-1,1,20,40)

        
#QDM        
        e1 = self.branch_e1(e1)
        r = self.branch_r(r)
        d = self.branch_d(d)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        h = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        h = h.view(-1, 1, 1, 800)
        #r = r.view(-1, 1, 1, 200)          
        
#DRN_e1       
        e1_At = torch.sigmoid(h)
        e1 = torch.cat((torch.mul(h1,e1_At),torch.mul(h2,e1_At),torch.mul(h3,e1_At),torch.mul(h4,e1_At)),1)
        e1 = self.bn_DRN_e1(e1)
        e1 = F.relu(e1)
        e1 = e1.view(-1,4,800)
        #e1_1 = e1_1.view(-1,5,200)
        e1 = self.sum_pool1(e1)

        h_MR = e1.view(-1,1,20,40)
            
#IEM
        
        e1 = self.branch1_e1(h_MR)
        r = self.branch1_r(r)
        d = self.branch1_d(d)
        shared = self.branch1_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate   
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        d = self.branch2_d(d)
        shared = self.branch2_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        d = self.branch3_d(d)
        shared = self.branch3_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch4_e1(e1)
        r = self.branch4_r(r)
        d = self.branch4_d(d)
        shared = self.branch4_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        entity = e1.view(-1, 2400)
        relation = r.view(-1, 2400)
        temporal = d.view(-1, 2400)
        
        e1 = self.linear_E(entity)
        full_rel = self.linear_R(relation)
        d = self.linear_d(temporal)
        e1 = F.tanh(e1)
        full_rel = F.tanh(full_rel)
        d = F.tanh(d) 
        r_ds = torch.mul(torch.mul(e1,full_rel),d)
        r_ds = r_ds.view(-1,1,56000)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,800)
        x = torch.mul(x, rnt)

        return torch.sum(x *rhs, 1, keepdim = True)

	
    def pretrain(self, x):
        
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rnt = self.embeddings[6](x[:, 3] // self.time_granularity)
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity)
        
        d = time.view(-1,1, 20,40) 
#MRN_e1        
        lhs1 = self.embeddings[3](x[:, 0])
        lhs2 = self.embeddings[4](x[:, 0])
        lhs3 = self.embeddings[5](x[:, 0])
       
        h1 = lhs.view(-1,1,1,800)
        h2 = lhs1.view(-1,1,1,800)
        h3 = lhs2.view(-1,1,1,800)
        h4 = lhs3.view(-1,1,1,800)
       
        h = torch.cat((h1,h2,h3,h4),1)
        h = self.conv21(h)
        h = self.bn_MRN_e1(h)
        h = F.relu(h)
        h = self.conv22(h)  
        e1 = h.view(-1,1,20,40)  
       
        r = rel.view(-1,1,20,40)
        
#QDM        
        e1 = self.branch_e1(e1)
        r = self.branch_r(r)
        d = self.branch_d(d)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        h = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        h = h.view(-1, 1, 1, 800)        
        
#DRN_e1       
        e1_At = torch.sigmoid(h)
        e1 = torch.cat((torch.mul(h1,e1_At),torch.mul(h2,e1_At),torch.mul(h3,e1_At),torch.mul(h4,e1_At)),1)
        e1 = self.bn_DRN_e1(e1)
        e1 = F.relu(e1)
        e1 = e1.view(-1,4,800)
        #e1_1 = e1_1.view(-1,5,200)
        e1 = self.sum_pool1(e1)

        h_MR = e1.view(-1,1,20,40)   

#IEM
        
        e1 = self.branch1_e1(h_MR)
        r = self.branch1_r(r)
        d = self.branch1_d(d)
        shared = self.branch1_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        d = self.branch2_d(d)
        shared = self.branch2_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        d = self.branch3_d(d)
        shared = self.branch3_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch4_e1(e1)
        r = self.branch4_r(r)
        d = self.branch4_d(d)
        shared = self.branch4_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        entity = e1.view(-1, 2400)
        relation = r.view(-1, 2400)
        temporal = d.view(-1, 2400)
        
        e3 = self.linear_E(entity)
        r3 = self.linear_R(relation)
        d3 = self.linear_d(temporal)
        e3 = F.tanh(e3)
        r3 = F.tanh(r3)
        d3 = F.tanh(d3) 
        r_ds = torch.mul(torch.mul(e3,r3),d3)
        r_ds = r_ds.view(-1,1,56000)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,800)
        x = torch.mul(x, rnt)

        to_score = self.embeddings[0].weight

        return (
                       x @ to_score.t()
               ),(entity,relation,temporal,rhs), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight


    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rnt = self.embeddings[6](x[:, 3] // self.time_granularity)
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity) 
        #rnt = self.embeddings[3](x[:, 1])
        
        d = time.view(-1,1,20,40) 
#MRN_e1        
        lhs1 = self.embeddings[3](x[:, 0])
        lhs2 = self.embeddings[4](x[:, 0])
        lhs3 = self.embeddings[5](x[:, 0])
       
        h1 = lhs.view(-1,1,1,800)
        h2 = lhs1.view(-1,1,1,800)
        h3 = lhs2.view(-1,1,1,800)
        h4 = lhs3.view(-1,1,1,800)
       
        h = torch.cat((h1,h2,h3,h4),1)
        h = self.conv21(h)
        h = self.bn_MRN_e1(h)
        h = F.relu(h)
        h = self.conv22(h)  
        e1 = h.view(-1,1,20,40)     
       
        r = rel.view(-1,1,20,40)

        
#QDM        
        e1 = self.branch_e1(e1)
        r = self.branch_r(r)
        d = self.branch_d(d)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        h = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        h = h.view(-1, 1, 1, 800)       
        
#DRN_e1       
        e1_At = torch.sigmoid(h)
        e1 = torch.cat((torch.mul(h1,e1_At),torch.mul(h2,e1_At),torch.mul(h3,e1_At),torch.mul(h4,e1_At)),1)
        e1 = self.bn_DRN_e1(e1)
        e1 = F.relu(e1)
        e1 = e1.view(-1,4,800)
        #e1_1 = e1_1.view(-1,5,200)
        e1 = self.sum_pool1(e1)

        h_MR = e1.view(-1,1,20,40)
            
#IEM
        
        e1 = self.branch1_e1(h_MR)
        r = self.branch1_r(r)
        d = self.branch1_d(d)
        shared = self.branch1_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        d = self.branch2_d(d)
        shared = self.branch2_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        d = self.branch3_d(d)
        shared = self.branch3_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch4_e1(e1)
        r = self.branch4_r(r)
        d = self.branch4_d(d)
        shared = self.branch4_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        right = self.embeddings[0].weight
        
        entity = e1.view(-1, 2400)
        relation = r.view(-1, 2400)
        temporal = d.view(-1, 2400)
        
        e3 = self.linear_E(entity)
        r3 = self.linear_R(relation)
        d3 = self.linear_d(temporal)
        e3 = F.tanh(e3)
        r3 = F.tanh(r3)
        d3 = F.tanh(d3) 
        r_ds = torch.mul(torch.mul(e3,r3),d3)
        r_ds = r_ds.view(-1,1,56000)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,800)
        x = torch.mul(x, rnt)

        regularizer = (entity, relation,temporal,rhs)
        return ((
               x @ right.t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )



    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rnt = self.embeddings[6](queries[:, 3] // self.time_granularity)
        time = self.embeddings[2](queries[:, 3] // self.time_granularity) 
        
        d = time.view(-1,1,20,40) 
#MRN_e1        
        lhs1 = self.embeddings[3](queries[:, 0])
        lhs2 = self.embeddings[4](queries[:, 0])
        lhs3 = self.embeddings[5](queries[:, 0])
       
        h1 = lhs.view(-1,1,1,800)
        h2 = lhs1.view(-1,1,1,800)
        h3 = lhs2.view(-1,1,1,800)
        h4 = lhs3.view(-1,1,1,800)
       
        h = torch.cat((h1,h2,h3,h4),1)
        h = self.conv21(h)
        h = self.bn_MRN_e1(h)
        h = F.relu(h)
        h = self.conv22(h)  
        e1 = h.view(-1,1,20,40)     
       
        r = rel.view(-1,1,20,40)
        
#QDM        
        e1 = self.branch_e1(e1)
        r = self.branch_r(r)
        d = self.branch_d(d)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        h = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        h = h.view(-1, 1, 1, 800)    
        
#DRN_e1       
        e1_At = torch.sigmoid(h)
        e1 = torch.cat((torch.mul(h1,e1_At),torch.mul(h2,e1_At),torch.mul(h3,e1_At),torch.mul(h4,e1_At)),1)
        e1 = self.bn_DRN_e1(e1)
        e1 = F.relu(e1)
        e1 = e1.view(-1,4,800)
        #e1_1 = e1_1.view(-1,5,200)
        e1 = self.sum_pool1(e1)

        h_MR = e1.view(-1,1,20,40)   

#IEM
        
        e1 = self.branch1_e1(h_MR)
        r = self.branch1_r(r)
        d = self.branch1_d(d)
        shared = self.branch1_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate  
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        d = self.branch2_d(d)
        shared = self.branch2_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        d = self.branch3_d(d)
        shared = self.branch3_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch4_e1(e1)
        r = self.branch4_r(r)
        d = self.branch4_d(d)
        shared = self.branch4_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        entity = e1.view(-1, 2400)
        relation = r.view(-1, 2400)
        temporal = d.view(-1, 2400)
        
        e3 = self.linear_E(entity)
        r3 = self.linear_R(relation)
        d3 = self.linear_d(temporal)
        e3 = F.tanh(e3)
        r3 = F.tanh(r3)
        d3 = F.tanh(d3) 
        r_ds = torch.mul(torch.mul(e3,r3),d3)
        r_ds = r_ds.view(-1,1,56000)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,800)
        x = torch.mul(x, rnt)
       
        return x

    def get_lhs_queries(self, queries: torch.Tensor):
        rhs = self.embeddings[0](queries[:, 2])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity)
        rnt = self.embeddings[6](queries[:, 3] // self.time_granularity)
        
        d = time.view(-1,1,20,40) 
#MRN_e1        
        rhs1 = self.embeddings[3](queries[:, 2])
        rhs2 = self.embeddings[4](queries[:, 2])
        rhs3 = self.embeddings[5](queries[:, 2])
       
        h1 = rhs.view(-1,1,1,800)
        h2 = rhs1.view(-1,1,1,800)
        h3 = rhs2.view(-1,1,1,800)
        h4 = rhs3.view(-1,1,1,800)
       
        h = torch.cat((h1,h2,h3,h4),1)
        h = self.conv21(h)
        h = self.bn_MRN_e1(h)
        h = F.relu(h)
        h = self.conv22(h)  
        e1 = h.view(-1,1,20,40)     
       
        r = rel.view(-1,1,20,40)
        
#QDM      
        e1 = self.branch_e1(e1)
        r = self.branch_r(r)
        d = self.branch_d(d)
        
        shared = torch.zeros(e1.shape).cuda()
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        h = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate
        
        h = h.view(-1, 1, 1, 800)        
        
#DRN_e1       
        e1_At = torch.sigmoid(h)
        e1 = torch.cat((torch.mul(h1,e1_At),torch.mul(h2,e1_At),torch.mul(h3,e1_At),torch.mul(h4,e1_At)),1)
        e1 = self.bn_DRN_e1(e1)
        e1 = F.relu(e1)
        e1 = e1.view(-1,4,800)
        e1 = self.sum_pool1(e1)

        h_MR = e1.view(-1,1,20,40)
            
#IEM
        
        e1 = self.branch1_e1(h_MR)
        r = self.branch1_r(r)
        d = self.branch1_d(d)
        shared = self.branch1_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch2_e1(e1)
        r = self.branch2_r(r)
        d = self.branch2_d(d)
        shared = self.branch2_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch3_e1(e1)
        r = self.branch3_r(r)
        d = self.branch3_d(d)
        shared = self.branch3_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 
        
        e1 = self.branch4_e1(e1)
        r = self.branch4_r(r)
        d = self.branch4_d(d)
        shared = self.branch4_s(shared) 
        
        e1_fuse_gate = self.sa1(e1 - shared)
        e1_fuse_gate = F.relu(e1_fuse_gate)
        r_fuse_gate= self.sa1(r- shared)
        r_fuse_gate = F.relu(r_fuse_gate)
        d_fuse_gate = self.sa1(d- shared)
        d_fuse_gate = F.relu(d_fuse_gate)
        shared = shared + (e1 - shared) * e1_fuse_gate + (r - shared) * r_fuse_gate + (d - shared) * d_fuse_gate
        
        e1_distribute_gate = self.sa1(shared - e1)
        e1_distribute_gate = F.relu(e1_distribute_gate)
        r_distribute_gate = self.sa1(shared - r)
        r_distribute_gate = F.relu(r_distribute_gate)
        d_distribute_gate = self.sa1(shared - d)
        d_distribute_gate = F.relu(d_distribute_gate)
        e1 = e1 + (shared - e1) * e1_distribute_gate
        r = r + (shared- r) * r_distribute_gate
        d = d + (shared- d) * d_distribute_gate 

        
        entity = e1.view(-1, 2400)
        relation = r.view(-1, 2400)
        temporal = d.view(-1, 2400)
        
        e3 = self.linear_E(entity)
        r3 = self.linear_R(relation)
        d3 = self.linear_d(temporal)
        e3 = F.tanh(e3)
        r3 = F.tanh(r3)
        d3 = F.tanh(d3) 
        r_ds = torch.mul(torch.mul(e3,r3),d3)
        r_ds = r_ds.view(-1,1,56000)
        x = self.sum_pool(r_ds)
        x = F.normalize(x,p=2,dim=2)
        x = x.view(-1,800)
        x = torch.mul(x, rnt)    

        return x