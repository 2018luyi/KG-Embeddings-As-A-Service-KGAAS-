import config
import models
import tensorflow as tf
import numpy as np
import json
import sys
import pdb
import os

main_dir = '../../../Documents/KGAAS_data/'
con = config.Config()

#process graph and create input files
def process_graph(kg):
    data_dir = main_dir+kg+'_data/'
    try:
        os.mkdir(data_dir)
    except OSError:
        print(data_dir)
 
    entities = set()
    relations = set()
    edgelist = list()

    with open(main_dir+kg,'r') as f:
        for line in f:
            items = line.strip().split(' ') 
            entities.add(items[0])
            entities.add(items[2])
            relations.add(items[1])
            edgelist.append((items[0],items[2],items[1]))

    entities = list(entities)
    relations = list(relations)
    edgelist = list(set(edgelist))
    entities_dict = dict()
    relations_dict = dict()

    with open(data_dir+'entity2id.txt','w') as f:
        f.write('{}\n'.format(len(entities)))
        for ii, item in enumerate(entities):
            f.write('{}\t{}\n'.format(item,ii))
            entities_dict[item] = ii         
     
    with open(data_dir+'relation2id.txt','w') as f:
        f.write('{}\n'.format(len(relations)))
        for ii, item in enumerate(relations):
            f.write('{}\t{}\n'.format(item,ii))
            relations_dict[item] = ii

    with open(data_dir+'train2id.txt','w') as f:
        f.write('{}\n'.format(len(edgelist)))
        for item in edgelist:
            f.write('{}\t{}\t{}\n'.format(entities_dict[item[0]],entities_dict[item[1]], relations_dict[item[2]]))
            
    
    return data_dir


def generate_ke_embeddings(args_dict):
    data_dir = process_graph(args_dict['knowledge_graph'])
    pdb.set_trace()
    con.set_in_path(graph_dir)
    con.set_work_threads(1)
    con.set_train_times(args_dict['train_time'])
    con.set_nbatches(args_dict['n_batches'])
    con.set_alpha(0.1)
    con.set_bern(0)
    con.set_margin(1)
    con.set_dimension(args_dict['dimension'])
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method(args_dict['optimizer'])
    con.set_export_files(data_dir+args_dict['model']+"_model.vec.tf", 0)
    con.set_out_files(data_dir+args_dict['model']+"_embedding.vec.json")
    con.init()
    con.set_model(models.args_dict['model'])
    con.run()



#TODO 
#def generate_walking_embeddings(): #walking RDF and OWL
#def generate_poincare_embeddings():
#def generate_rgcn_embeddings(): #relational-GCN

#def predict_most_similar(input_entities):
#def predict_links(edglist_)
#def housekepping(): remove all generated intermediate files


def main():
    args = open(sys.argv[1]).read().splitlines()
    args_dict = {it.split(':')[0]:it.split(':')[1] for it in args}
    #pdb.set_trace()
    generate_embeddings(args_dict)
    pdb.set_trace() 
    

if __name__ == '__main__':
    main()
