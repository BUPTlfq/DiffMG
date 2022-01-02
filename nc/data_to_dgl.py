import os
import sys
import numpy as np
import torch
import pickle as pkl
import scipy.sparse as sp
import dgl
cstr_nc = {
    "DBLP": [1, 4],
    "IMDB": [0, 2, 4]
}

categorys = {
    "DBLP": 'Author',
    "IMDB": 'Movie'
}

node_types = {
    "DBLP": ['Paper', 'Author', 'Conference'],
    "IMDB": ['Movie', 'Director', 'Actor']
}
edge_types = {
    "DBLP": [{'src': 'Paper', 'dst': 'Author'}, {'src': 'Author', 'dst': 'Paper'},
             {'src': 'Paper', 'dst': 'Conference'}, {'src': 'Conference', 'dst': 'Paper'}],
    "IMDB": [{'src': 'Movie', 'dst': 'Director'}, {'src': 'Director', 'dst': 'Movie'},
             {'src': 'Movie', 'dst': 'Actor'}, {'src': 'Actor', 'dst': 'Movie'}]
}

split_label_list = ['train', 'valid', 'test']


def main(dataset):
    prefix = os.path.join("./data/", dataset)
    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pkl.load(f)
        f.close()

    # <========= change every type node_id start from zero in dgl =======>
    node_type = node_types[dataset]
    start_id_dict = {}
    a = np.unique(list(edges[0].tocoo().row) + list(edges[2].tocoo().row))
    start_id_dict[node_type[0]] = np.min(a)

    b = np.unique(edges[0].tocoo().col)
    start_id_dict[node_type[1]] = np.min(b)

    c = np.unique(edges[2].tocoo().col)
    start_id_dict[node_type[2]] = np.min(c)
    assert (a.shape[0] + b.shape[0] + c.shape[0] == edges[0].shape[0])

    meta_graphs = {}  # dict will to dgl
    for edge in range(len(edges)):
        relation_graph = edges[edge].tocoo()
        src_dst = edge_types[dataset][edge]
        meta_graphs[(src_dst['src'], src_dst['src']+'_to_'+src_dst['dst'], src_dst['dst'])] = (
            torch.tensor(relation_graph.row-start_id_dict[src_dst['src']]), torch.tensor(relation_graph.col-start_id_dict[src_dst['dst']]))
        # construct edge
    g = dgl.heterograph(meta_graphs)

    with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
        node_feature = pkl.load(f)
        f.close()
    # add node feature
    for key, value in start_id_dict.items():
        g.nodes[key].data['feature'] = torch.tensor(node_feature[value:(value+g.num_nodes(key))])

    # you can get node feature use g.nodes[key].data['feature']

    with open(os.path.join(prefix, "labels.pkl"), "rb") as f:
        label_list = pkl.load(f)
        f.close()
    category = categorys[dataset]
    category_label = torch.full((g.num_nodes(category), ), -1, dtype=torch.int32)

    for split in range(len(split_label_list)):
        index, label = get_index_label(label_list[split], start_id_dict[category])
        index = torch.LongTensor(index)
        label = torch.tensor(label)
        category_label[index] = label
        mask = torch.full((g.num_nodes(category), ), False)
        mask[index] = True
        g.nodes[category].data[split_label_list[split]+'_mask'] = mask
    g.nodes[category].data['category_label'] = category_label

    '''
        concat train and valid and test label.
        you can get bool mask train_mask = g.nodes[category_name].data['train_mask']
        you can get the train label use g.nodes[category_name].data['label'][train_mask]
    '''
    if not os.path.exists('dgl_graph'):
        os.makedirs('dgl_graph')
        save_path = os.path.join("dgl_graph", dataset+'.bin')
        dgl.save_graphs(save_path, g)
    else:
        save_path = os.path.join("dgl_graph", dataset + '.bin')
        dgl.save_graphs(save_path, g)
    # dgl.save_graphs()


def get_index_label(label_list, start_id):
    label_list = np.array(label_list)
    index = label_list[:, 0] - start_id
    label = label_list[:, 1]
    return index, label


if __name__ == "__main__":
    main("DBLP")
    main("IMDB")