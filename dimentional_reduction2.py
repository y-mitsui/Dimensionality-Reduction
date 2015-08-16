from dA import dA
import numpy
from logistic_sgd import load_data
import theano
import theano.tensor as T
import sys
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def test_DimentionalReduction(learning_rate=0.1,training_epochs=15,dataset='mnist.pkl.gz',batch_size=20):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
   

    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=2
    )
    
    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    
    x = T.matrix('x')
    hidden_values_function = da.get_hidden_values2(x)
    
    result_function = theano.function(inputs=[x],outputs=hidden_values_function)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','w']
    n = 0
    for x,y in zip(result_function(train_set_x.get_value()),train_set_y.eval()):
        if y < len(colors):
            plt.scatter(x[0], x[1],c=colors[y])
            n += 1
            
        if n > 2000:
            break
        
    plt.show()
    n = 0
    pca = PCA(n_components=2)
    for x,y in zip(pca.fit_transform(train_set_x.get_value()),train_set_y.eval()):
        if y < len(colors):
            plt.scatter(x[0], x[1],c=colors[y])
            n += 1
            
        if n > 2000:
            break
        
    plt.show()
if __name__ == '__main__':
    test_DimentionalReduction()
