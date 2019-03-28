import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  now = X.dot(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    sigma_tmp = 0.0
    yi_tmp = 0.0
    for j in range(num_classes):
      if j == y[i]:
        yi_tmp = np.exp(now[i, j])
      sigma_tmp += np.exp(now[i, j])
    dW[:, y[i]] += -np.log(np.e) * (sigma_tmp-yi_tmp) / sigma_tmp * X[i]
    for j in range(num_classes):
      if j != y[i]:
        dW[:, j] += np.log(np.e) / sigma_tmp * np.exp(now[i, j]) * X[i]
    loss += -np.log(yi_tmp/sigma_tmp)
  dW /= num_train
  dW += reg * W
  loss /= num_train
  loss += reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  now = X.dot(W)
  now_exp = np.exp(now)
  now_sum = np.sum(now_exp, axis=1)
  # print(now_sum)
  arr = np.arange(X.shape[0])
  now_tmp=(np.exp(now[arr, y[arr]]) / now_sum)
  # print(now_tmp)
  loss += -np.sum(np.log(now_tmp))
  loss /= X.shape[0]
  loss += reg*np.sum(W*W)
  margin = now_exp / now_sum.reshape(X.shape[0], 1)
  margin[arr, y] -= 1
  dW = X.T.dot(margin)
  dW /= X.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

