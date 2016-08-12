import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  correct_scores = []
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    correct_scores.append(scores[y[i]])
    count_above_zero = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count_above_zero += 1
        dW[:,j] += X[i]
    dW[:,y[i]] += -count_above_zero * X[i] 
      
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += 0.5 * reg * 2 * W 

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


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
  train_num = X.shape[0]

  mul = X.dot(W)
  correct_scores = mul[range(X.shape[0]),y]
  mul_subtracted = mul - correct_scores[:, np.newaxis] + 1
    
  #make the correct class scores to zero for caculationg loss
  mul_subtracted[range(X.shape[0]), y] = 0
    
  #use a mask to perform maximum function
  mask = mul_subtracted > 0 
  mul_subtracted *=  mask
  #mul_subtracted = np.maximum(np.zeros(mul_subtracted.shape), mul_subtracted)

    
  loss = (np.sum(mul_subtracted) / train_num) + 0.5 * reg * np.sum(W * W)
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################


  ### This method extends calculation dimension to 3-d,
  ### then sum over the trainning-set axis to get final results.
  ### However, it turns out the extend-and-sum-over process can be
  ### simplified by a matrix multiplication, which is much faster.
  #repeated_train = X[:,:,np.newaxis] * mask[:,np.newaxis,:]
  #print "shape of x is %s" % list(X.shape)
  #mul_train = -X * np.sum(mask, axis=1)[:, np.newaxis]
  #repeated_train[[range(X.shape[0])],:,y] = mul_train
  #dW = np.sum(repeated_train, axis=0)
    
  #transfer bool array into integers 0 and 1  
  mask = np.array(mask, dtype=int)
    
  #count how many wrong-class scores are higher than right-class score
  #then put the number into the corresponding element 
  mask[range(X.shape[0]), y] = -np.sum(mask, axis=1)
  
  dW = X.T.dot(mask)
  dW = dW / train_num
  dW += 0.5 * reg * 2 * W 
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
