"""
Tensorflow implementation of Convolutioanl sparse autoenocer, 
also known as Winner-Takes-All autoencoder.

Reference: 
  [1] https://arxiv.org/pdf/1409.2752.pdf
  
Author: Arash Saber Tehrani
"""

import tensorflow as tf
import tflearn

class sparseAE:
    """
    Args:
        session: the active session
        kernel_size: number of outputs for the three convolutional layers
        
    """
    def __init__(self, session, kernel_size=None, name='sparseAE'):
        if kernel_size is None:
            kernel_size = [128, 128, 16]
        self.name = name
        self.kernel_size = kernel_size

        self.sess = session

    def encoder(self, input_shape):
        net = tflearn.layers.core.input_data(shape=input_shape, name='input')
        net = self._conv(net, self.kernel_size[0], [5, 5], scope='conv1')
        net = self._conv(net, self.kernel_size[1], [5, 5], scope='conv2')
        net = self._conv(net, self.kernel_size[2], [5, 5], scope='conv3')
        return net

    def decoder(self, net):
        with tf.variable_scope('decoder'):
            y = self._deconv(net, nb_filter=1, filter_size=[11, 11])
        return y

    def build_model(self, input_shape, LR=1e-3, optimizer='adam', tb_verbose=3):
        """
        Build the sparseAE
        :param input_shape: 1Darray, shape of the input
        :param LR: scalar, learning rate
        :param optimizer: string, optimizer to use
        :param tb_verbose: int
        :return: tflearn dnn object
        """
        net = self.encoder(input_shape)
        self.sparse_rep = net
        net, _ = self.spatial_sparsity(net)
        net = self.decoder(net)
        net = tflearn.layers.estimator.regression(net,
                                                  optimizer=optimizer,
                                                  learning_rate=LR,
                                                  loss='mean_square',
                                                  name='targets')
        self.model = tflearn.DNN(net, tensorboard_dir='logs', tensorboard_verbose=tb_verbose, session=self.sess)

    def train(self, x, val_x, n_epochs=10,
              batch_size=100, snapshot_step=5000, show_metric=True):
        """
        Train the sparseAE
        :param x: input data to feed the network
        :param val_x: validation data
        :param n_epochs: int, number of epochs
        :param batch_size: int
        :param snapshot_step: int
        :param show_metric: boolean
        """
        self.sess.run(tf.global_variables_initializer())
        self.model.fit({'input': x}, {'targets': x}, n_epoch=n_epochs,
                       batch_size=batch_size,
                       validation_set=({'input': val_x}, {'targets': val_x}),
                       snapshot_step=snapshot_step,
                       show_metric=show_metric,
                       run_id='SparseAE')

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

    def _conv(self, input, nb_filter, filter_size, strides=None, scope='conv'):
        if strides is None:
            strides=[1,1,1,1]
        output = tflearn.layers.conv.conv_2d(input, nb_filter, filter_size,
                                             activation='relu', scope=scope,
                                             strides=strides,
                                             padding='same',
                                             bias=True,
                                             weights_init=tflearn.initializations.xavier(uniform=False),
                                             bias_init=tflearn.initializations.xavier(uniform=False))

        return output

    def _deconv(self, input, nb_filter, filter_size, strides=None, scope='deconv'):
        if strides is None:
            strides=[1,1]
        output = tf.contrib.layers.conv2d_transpose(input,
                                                    num_outputs=nb_filter,
                                                    kernel_size=filter_size,
                                                    stride=strides,
                                                    padding='same',
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                        uniform=False),
                                                    biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                    activation_fn=tf.nn.relu,
                                                    scope=scope)

        return output

    def spatial_sparsity(self, net):
        shape = tf.shape(net)
        b = shape[0]
        c = shape[3]

        net_t = tf.transpose(net, [0, 3, 1, 2])  # b, c, h, w
        net_r = tf.reshape(net_t, tf.stack([b, c, -1]))  # b, c, h*w

        th, _ = tf.nn.top_k(net_r, 1)  # b, c, 1
        th_r = tf.reshape(th, tf.stack([b, 1, 1, c]))  # b, 1, 1, c
        drop = tf.where(net < th_r,
                         tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))

        # spatially dropped and top element (winners)
        return net * drop, tf.reshape(th, tf.stack([b, c]))  # b, c


if __name__ == '__main__':
    """
    let us test it on MNIST database"""
    import tflearn.datasets.mnist as mnist

    X, _, valX, _ = mnist.load_data(one_hot=True)
    X = X[:500].reshape([-1, 28, 28, 1])
    valX = valX[:100].reshape([-1, 28, 28, 1])

    with tf.Session() as sess:
        ae = sparseAE(sess)
        ae.build_model([None,28,28,1])
        # train the Autoencoder
        ae.train(X, valX, n_epochs=1) # valX for validation
        # compute the output for a certain input
        out = ae.model.predict(X[0].reshape([-1, 28, 28, 1]))
        print(out)
        # get the weights of a certain layer
        vars = tflearn.get_layer_variables_by_name('conv3') # in this case, it is the learned features
        W = ae.model.get_weights(vars[0])
        print(W.shape)
        # get output of encoder for certain input
        m2 = tflearn.DNN(ae.sparse_rep, session=sess)
        print(m2.predict(X[0].reshape([-1, 28, 28, 1]) ))
        # save and load the model
        ae.save('./sparseAE.tflearn')
        ae.load('./sparseAE.tflearn')
