import json

def get_data(dataset='vrd'):
    if dataset == 'vrd':
        data = json.load(open('../../data/vrd/train.json'))
        return data['ind_to_class'], data['ind_to_predicate'], data['data']
    else:
        data = None
        return data

def prepare_data(classes, predicates, data, cls_file, pred_file, node_file, edge_file):
    # write nodes: class and predicate
    with open(node_file, 'w') as f :
        with open(cls_file, 'w') as cls_f:
            with open(pred_file, 'w') as pred_f:
                for i in range(len(classes)):
                    f.write('%d %s\n'%(i, 'o'))
                    cls_f.write('%s %d\n'%(classes[i].replace(' ', '_'), i))
                for i in range(len(predicates)):
                    f.write('%d %s\n'%(i+len(classes), 'p'))
                    pred_f.write('%s %d\n' % (predicates[i].replace(' ', '_'), i+len(classes)))

    with open(edge_file, 'w') as f:
        for ele in data:
            rels = ele['relations']
            labels = ele['labels']
            for rel in rels:
                o1, o2 = labels[rel[0]], labels[rel[1]]
                f.write('%d %d %d\n'%(o1, rel[2]+len(classes), o2))
                # f.write('%d %d\n'%(rel[2]+len(classes), o2))

if __name__ == '__main__':
    cls, pred, data  = get_data('vrd')
    path='./src-cikm/vrd/'
    prepare_data(cls, pred, data, path+'conf_category.txt', path+'word_category.txt', path+'nodes.txt', path+'edges.txt')