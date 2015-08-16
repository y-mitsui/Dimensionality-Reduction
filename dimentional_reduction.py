import sys
import numpy
from logistic_sgd import load_data
from SdA import SdA
from logistic_sgd import load_data
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt


def test_DimentionalReduction(dataset='mnist.pkl.gz', pretraining_epochs=15, pretrain_lr=0.001, batch_size=100):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    numpy_rng = numpy.random.RandomState(89677)
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[300, 50, 2],
        n_outs=2
    )
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    corruption_levels = [0., 0., 0.]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    target = train_set_x.get_value()
    for dA_layer in sda.dA_layers:
        hidden_values_function = dA_layer.get_hidden_values2(sda.x)
        result_function = theano.function(inputs=[sda.x],outputs=hidden_values_function)
        target = result_function(target)
        print target

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','w']
    n = 0
    for x,y in zip(target,train_set_y.eval()):
        if y < len(colors):
            plt.scatter(x[0], x[1],c=colors[y])
            n += 1
            
        if n > 2000:
            break
        
    plt.show()

if __name__ == '__main__':
    test_DimentionalReduction()
