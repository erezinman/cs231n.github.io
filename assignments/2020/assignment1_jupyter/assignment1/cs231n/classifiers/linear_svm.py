from builtins import range

import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                for m in range(W.shape[0]):
                    # n == j
                    dW[m, j] += X[i, m]
                    # n == y[i]
                    dW[m, y[i]] -= X[i, m]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W
    margins = np.maximum(0, scores - scores[np.arange(len(y)), y].reshape(-1, 1) + 1)
    margins[np.arange(margins.shape[0]), y] = 0
    data_loss = (1 / X.shape[0]) * np.sum(margins)
    reg_loss = reg * np.sum(W ** 2)
    loss = data_loss + reg_loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Option 1 (takes too-much memory and hence )
    # dscores = np.zeros((*margins.shape, *W.shape), dtype=float)
    # dscores[:, np.arange(margins.shape[1]), :, np.arange(margins.shape[1])] += X
    # dscores[np.arange(len(y)), :, :, y] -= X[:, np.newaxis, :]
    # dscores[margins <= 0] = 0
    # dW[:, :] = np.sum(dscores, axis=(0, 1))

    # Option 2 (better)
    indicators = (margins > 0).astype(float)
    transfer = np.diag(np.ones(W.shape[1]))[y]
    indicator_count = np.sum(indicators, axis=1).reshape((-1, 1))
    dW[:, :] = X.T @ (indicators - transfer * indicator_count)

    # For both options
    dW = dW / X.shape[0] + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


if __name__ == '__main__':
    import pickle

    with open('/tmp/tmp.pkl', 'rb') as f:
         W, X_dev, y_dev = pickle.load(f)

    svm_loss_vectorized(W, X_dev, y_dev, 0.000005)