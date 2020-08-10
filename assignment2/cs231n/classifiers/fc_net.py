from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_layer, hidden_layer_cache = affine_relu_forward(X, W1, b1)
        scores, cache = affine_forward(hidden_layer, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5*self.reg*np.sum(W1*W1) + 0.5*self.reg*np.sum(W2*W2)

        loss = data_loss + reg_loss

        dhidden, dW2, db2 = affine_backward(dscores, cache)

        _, dW1, db1 = affine_relu_backward(dhidden, hidden_layer_cache)

        dW2 += self.reg*W2
        dW1 += self.reg*W1

        grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        def weight_init(weight_scale, dim1, dim2):
            """
            Helper function to set initial weights
            """
            return weight_scale * np.random.randn(dim1, dim2)

        input_size = input_dim

        for i in range(len(hidden_dims)):
            weight_key = f'W{i+1}'              # Indexing from 1 in keys, bias
            bias_key = f'b{i+1}'

            self.params[weight_key] = weight_init(weight_scale, input_size, hidden_dims[i])
            self.params[bias_key] = np.zeros(hidden_dims[i])

            input_size = hidden_dims[i]

        # Last Output layer
        weight_key = f'W{self.num_layers}'
        bias_key = f'b{self.num_layers}'
        self.params[weight_key] = weight_init(weight_scale, input_size, num_classes)
        self.params[bias_key] = np.zeros(num_classes)

        if self.normalization is not None:
            # batch norm parameters
            for i in range(self.num_layers-1):
                self.params[f'gamma{i+1}'] = np.ones(hidden_dims[i])
                self.params[f'beta{i+1}'] = np.zeros(hidden_dims[i])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = []
        input = X
        output = None
        gamma, beta, bn_params = None, None, None

        for layer in range(self.num_layers-1):      # Last layer is output
            weight = self.params[f'W{layer+1}']
            bias = self.params[f'b{layer+1}']

            if self.normalization is not None:
                gamma = self.params[f'gamma{layer+1}']
                beta = self.params[f'beta{layer+1}']
                bn_params = self.bn_params[layer]

            output, cache = affine_norm_relu_dropout_forward(input, weight, bias,
                                gamma, beta, bn_params, self.normalization,
                                self.use_dropout, self.dropout_param)

            caches.append(cache)        # Store cache for backward pass
            input = output              # This output becomes next layer's input

        # Final output layer
        final_weight = self.params[f'W{self.num_layers}']
        final_bias = self.params[f'b{self.num_layers}']
        scores, final_cache = affine_forward(input, final_weight, final_bias)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward propogate for final layer
        data_loss, dscores = softmax_loss(scores, y)
        dhidden_input, dW_final, db_final = affine_backward(dscores, final_cache)

        # Calculate dW for final layer
        final_wkey = f'W{self.num_layers}'
        final_bkey = f'b{self.num_layers}'
        wfinal = self.params[final_wkey]
        dW_final += self.reg * wfinal

        grads = {
            final_wkey: dW_final,
            final_bkey: db_final,
        }

        # Initial reg loss
        reg_loss = 0.5 * self.reg * np.sum(wfinal*wfinal)

        # Now do for all hidden layers
        for layer in reversed(range(self.num_layers-1)):
            # Just setup some variables
            wkey = f'W{layer+1}'
            bkey = f'b{layer+1}'
            W = self.params[wkey]

            reg_loss += 0.5 * self.reg * np.sum(W*W)

            dhidden_input, dW, db, dgamma, dbeta = affine_norm_relu_dropout_backward(dhidden_input, caches[layer],
                                                    self.normalization, self.use_dropout)

            if self.normalization is not None:
                grads[f'gamma{layer+1}'] = dgamma
                grads[f'beta{layer+1}'] = dbeta

            dW += self.reg * W

            grads[wkey] = dW
            grads[bkey] = db

        # Finally sum up loss
        loss = data_loss + reg_loss

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_norm_relu_dropout_forward(x, w, b, gamma, beta, bn_params, normalization, use_dropout, dropout_param):
    """
    Helper forward function for affine-norm-relu-dropout layers combination

    Returns: out, cache (Tuple of (fc_cache, n_cache, relu_cache, dropout_cache) )
    """
    n_cache, dropout_cache = None, None

    # Affine
    out, fc_cache = affine_forward(x, w, b)

    # Batch / Layer Normalization
    if normalization == 'batchnorm':
        out, n_cache = batchnorm_forward(out, gamma, beta, bn_params)
    elif normalization == 'layernorm':
        out, n_cache = layernorm_forward(out, gamma, beta, bn_params)

    # ReLU
    out, relu_cache = relu_forward(out)

    # Dropout
    if use_dropout:
        out, dropout_cache = dropout_forward(out, dropout_param)

    return out, (fc_cache, n_cache, relu_cache, dropout_cache)


def affine_norm_relu_dropout_backward(dout, cache, normalization, use_dropout):
    """
    Helper backward function for affine-norm-relu-dropout layers

    Returns: dx, dw, db, dgamma, dbeta
    """
    fc_cache, n_cache, relu_cache, dropout_cache = cache

    # Dropout
    if use_dropout:
        dout = dropout_backward(dout, dropout_cache)

    # ReLU
    dout = relu_backward(dout, relu_cache)

    # Normalization
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
        dout, dgamma, dbeta = batchnorm_backward_alt(dout, n_cache)
    elif normalization == 'layernorm':
        dout, dgamma, dbeta = layernorm_backward(dout, n_cache)

    # Affine
    dx, dw, db = affine_backward(dout, fc_cache)

    return dx, dw, db, dgamma, dbeta
