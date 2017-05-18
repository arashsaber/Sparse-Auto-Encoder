# Sparse-Auto-Encoder

Tensorflow (tflearn) implementation of Convolutioanl sparse autoenocer, 
also known as Winner-Takes-All autoencoder.

To use:

    ae = sparseAE(sess)
    ae.build_model([None,28,28,1])

train the Autoencoder
    ae.train(X, valX, n_epochs=1) # valX for validation

compute the output for a certain input

    out = ae.model.predict(X[0].reshape([-1, 28, 28, 1]))
    
get the weights of a certain layer
        
    vars = tflearn.get_layer_variables_by_name('conv3') # in this case, it is the learned features
    W = ae.model.get_weights(vars[0])

get output of encoder for certain input
    
    m2 = tflearn.DNN(ae.sparse_rep, session=sess)
    m2.predict(X[0].reshape([-1, 28, 28, 1]) )
 
save and load the model
        
    ae.save('./sparseAE.tflearn')
    ae.load('./sparseAE.tflearn')

Reference: 
[1] https://arxiv.org/pdf/1409.2752.pdf
