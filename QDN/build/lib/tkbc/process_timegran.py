import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import re
import sys
import argparse
#import matplotlib.pyplot as plt


import numpy as np

from collections import defaultdict

parser = argparse.ArgumentParser(
    description="TeLM"
)
parser.add_argument(
    '--tr', default=1, type=int,
    help="threshold of fact numbers for grouping time steps in YAGO and Wikdata"
)
parser.add_argument(
    '--dataset', type=str,
    help="YAGO or Wikdata"
)
args = parser.parse_args()

DATA_PATH = 'data/'

def prepare_dataset_rels(path, name, fact_count):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(type)\t(timestamp)\n
    Maps each entity, relation+type and timestamp to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, freq = set(), set(), defaultdict(int)
    total_facts = 0
    n=0
    year_list=[]
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        
        for line in to_read.readlines():
            n+=1
            line = line.split('\t')
            if line[3][0]=='-':
                start = -int(line[3].split('-')[1])
                year_list.append(start)
            else:
                start = line[3].split('-')[0]
                if start !='####':
                    start = start.replace('#', '0')
                    start = int(start)
                    year_list.append(start)


            if line[4][0]=='-':
                end = -int(line[4].split('-')[1])
                year_list.append(end)
            else:
                end = line[4].split('-')[0]
                if end !='####':
                    end = end.replace('#', '0')
                    end = int(end)
                    year_list.append(end)
                    
            entities.add(int(line[0]))
            entities.add(int(line[2]))
            relations.add(int(line[1]))
                              
        to_read.close()
        if f == 'train':
            year_list.sort()
        
            for year in year_list:
                freq[year]=freq[year]+1
        
            year_class=[]
            count=0
            for key in sorted(freq.keys()):
                count += freq[key]
                if count>= fact_count:
                    year_class.append(key)
                    count=0
            year_class[-1]=year_list[-1]
        
            year2id = dict()
            prev_year = year_list[0]
            i = 0
            for i, yr in enumerate(year_class): 
                year2id[(prev_year, yr)] = i
                prev_year = yr + 1





    print(f"{len(year2id.keys())} timestamps")
    

    n_relations = 2*len(relations)
    n_entities = max(entities)+1
    print("{} entities, {} relations over {} timestamps".format(n_entities, n_relations, len(year2id)))


    try:
        os.makedirs(os.path.join(DATA_PATH, name))
    except OSError as e:
        r = input(f"{e}\nContinue ? [y/n]")
        if r != "y":
            sys.exit()

    # write ent to id / rel to id
    ff = open(os.path.join(DATA_PATH, name, 'ts_id'), 'w+')
    for (x, i) in zip(year2id.keys(),year2id.values()):
        ff.write("{}\t{}\n".format(x, i))
    ff.close()
    out = open(Path(DATA_PATH) / name / 'ts_id.pickle', 'wb')
    pickle.dump(year2id, out)
    out.close()



    # map train/test/valid with the ids

    facts= {}
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        facts[f] = []
        for line in to_read.readlines():
            triple = line.strip().split('\t')
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

            
            facts[f].append([int(triple[0]), int(triple[1]), int(triple[2]), triple[3], triple[4]])
            if f == 'train':          
                if start_idx < 0:
                    examples.append([int(triple[0]), int(triple[1])+len(relations), int(triple[2]), end_idx])
                else:
                    examples.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])                
                if end_idx < 0:
                    examples.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                else:
                    examples.append([int(triple[0]), int(triple[1])+len(relations), int(triple[2]), end_idx])
            else:
                examples.append([int(triple[0]), int(triple[1]), int(triple[2]), triple[3], triple[4]])


        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        if f == 'train': 
            pickle.dump(np.array(examples).astype('uint64'), out)
        else:
            pickle.dump(np.array(examples), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = facts[f]
        for lhs, rel, rhs, ts, te in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts, te)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts, te)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()


    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        try:
            counters['rhs'][rhs] += 1
        except:
            a=1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()
    
    

if __name__ == "__main__":
    datasets = [args.dataset]
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset_rels(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'src_data', d
                ),
                d,
                fact_count=args.tr)
            
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise

