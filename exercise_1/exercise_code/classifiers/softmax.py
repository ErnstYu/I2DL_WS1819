"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    num_train, num_class = X.shape[0], W.shape[1]
    for i in range(num_train):
        scores = np.dot(X[i,:], W)
        exp_sum = np.sum(np.exp(scores))
        loss += -scores[y[i]] + np.log(exp_sum)
        
        for j in range(num_class):
            softmax_j = np.exp(scores[j]) / exp_sum
            if j == y[i]:
                dW[:,j] += (-1 + softmax_j) * X[i,:]
            else:
                dW[:,j] += softmax_j * X[i,:]
    
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW   = dW   / num_train + reg * W
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    num_train = X.shape[0]
    scores_exp = np.exp(np.dot(X, W))
    softmax = scores_exp / np.sum(scores_exp, axis=1).reshape(-1,1)
    loss = -np.sum(np.log(softmax[range(num_train), y] + 1e-12))
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    
    dS = softmax.copy()
    dS[range(num_train), y] += -1
    dW = np.dot(X.T, dS) / num_train + reg * W
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [2e-7, 2.5e-7, 3e-7, 3.5e-7, 4e-7, 4.5e-7, 5e-7]
    regularization_strengths = [6e3, 7e3, 8e3, 9e3, 1e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    for lr in learning_rates:
        for reg in regularization_strengths:
#            print(lr, reg)
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=3000)
            y_train_pred = softmax.predict(X_train)
            
            training_accuracy = np.mean(y_train == y_train_pred)
            y_val_pred = softmax.predict(X_val)
            val_accuracy = np.mean(y_val == y_val_pred)
            
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_softmax = softmax
            results[(lr, reg)] = training_accuracy, val_accuracy
            all_classifiers.append([softmax, val_accuracy])
    
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
