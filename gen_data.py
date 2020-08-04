import numpy as np 

def gen_data(nV=64, nall=10000, p=0.1, ntr=5000, nval=5000, graph_type='ring', std=1.5):
    # grid
    u = np.arange(nV)

    data = np.zeros((nall, nV))
    labels = np.zeros(nall)

    for idx in range(nall):
        tmp = np.random.uniform(size=nV) < p
        ## get indices of nodes have probability less than p
        p_indices = np.nonzero(tmp)[0]

        ## init data
        x = np.zeros(nV)
        
        for iu in p_indices:
            if graph_type == 'ring':
                relative_indices = np.mod(u-iu, nV)
                x += np.exp(-(relative_indices)**2/(2*std**2)) * (relative_indices < nV//2)
                    # (u > (iu-1)) # * np.sign(np.random.randn())
            elif graph_type == 'chain':
                x += np.exp(-(u-iu)**2/(2*std**2)) * (u>(iu-1))
        
        labels[idx] = 0

        if idx+1 > int(nall / 2):
            labels[idx] = 1
            ## reverse sample for another class
            x = x[-np.arange(1, len(x)+1)]
        
        data[idx] = x

    ## save data
    # to [nall, 1, nV]
    datas = np.expand_dims(data, 1).astype(np.float32)
    labels = labels.astype(np.int64)
    np.save('datas_{}.npy'.format(graph_type), datas)
    np.save('labels_{}.npy'.format(graph_type), labels)

    ## train-val split
    one_class_size_tr = ntr // 2
    train_datas = np.concatenate((datas[:one_class_size_tr], datas[nall-one_class_size_tr:]), axis=0)
    train_labels = np.concatenate((labels[:one_class_size_tr], labels[nall-one_class_size_tr:]))
    print(train_datas.shape)
    print('train class distribution: 0: {}, 1: {}'.format(ntr-np.sum(train_labels), np.sum(train_labels)))
    np.save('train_datas_{}.npy'.format(graph_type), train_datas)
    np.save('train_labels_{}.npy'.format(graph_type), train_labels)

    one_class_size_val = nval // 2
    # import pdb; pdb.set_trace()
    val_datas = np.concatenate((datas[one_class_size_tr:one_class_size_tr+one_class_size_val],
                            datas[nall-(one_class_size_tr+one_class_size_val):nall-one_class_size_tr]), axis=0)
    val_labels = np.concatenate((labels[one_class_size_tr:one_class_size_tr+one_class_size_val],
                            labels[nall-(one_class_size_tr+one_class_size_val):nall-one_class_size_tr]))
    print(val_datas.shape)
    print('val class distribution: 0: {}, 1: {}'.format(nval-np.sum(val_labels), np.sum(val_labels)))
    np.save('val_datas_{}.npy'.format(graph_type), val_datas)
    np.save('val_labels_{}.npy'.format(graph_type), val_labels)

    return datas, labels

def main(nV=64, nall=10000, p=0.1, ntr=5000, nval=5000, graph_type='ring'):
    gen_data(nV=nV, nall=nall, p=p, ntr=ntr, nval=nval, graph_type=graph_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('graph_type')
    parser.add_argument('--nV', type=int, default=64)
    parser.add_argument('--nall', type=int, default=10000)
    parser.add_argument('--p_thres', type=float, default=0.1)
    parser.add_argument('--ntr', type=int, default=5000)
    parser.add_argument('--nval', type=int, default=5000)
    # parser.add_argument('--nte', type=int, default=1000)

    args = parser.parse_args()

    main(nV=args.nV, nall=args.nall, p=args.p_thres, ntr=args.ntr, 
            nval=args.nval, graph_type=args.graph_type)
    
    print("Finish Generating Toy Data.")