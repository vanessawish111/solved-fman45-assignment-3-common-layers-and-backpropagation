Download Link: https://assignmentchef.com/product/solved-fman45-assignment-3-common-layers-and-backpropagation
<br>
<h1>1           Common Layers and Backpropagation</h1>

In this section you will implement forward and backward steps for a few common neural network layers.

Recall that the loss <em>L </em>: X × Y × W → R is a function of the network input <strong>x</strong>, the true value <strong>y </strong>and the network parameters <strong>w</strong>. When you train a neural net you have to evaluate the gradient with respect to all the parameters in the network. The algorithm to do this is called <em>backpropagation</em>.

<h2>1.1         Dense Layer</h2>

<strong>Introduction. </strong>Let a fixed, given layer in the network <em>f </em>= <em>σ </em>◦<em>` </em>: R<em><sup>m </sup></em>→ R<em><sup>n </sup></em>consist of a linear part <em>` </em>: R<em><sup>m </sup></em>→ R<em><sup>n </sup></em>and an activation function <em>σ </em>: R<em><sup>n </sup></em>→ R<em><sup>n</sup></em>.

We represent the linear part <em>` </em>by a weight matrix <strong>W </strong>and a bias vector <strong>b</strong>. Let <strong>x </strong>∈ R<em><sup>m </sup></em>be a data point that enters the given layer and let <strong>z </strong>= <em>`</em>(<strong>x</strong>) ∈ R<em><sup>n </sup></em>contain the values of the data point after having passed through the linear part of the layer. Then for all <em>i</em>,

<em>.                                                     </em>(1)

Using matrix notation we have <strong>z </strong>= <strong>Wx</strong>+<strong>b</strong>. Note that <strong>W </strong>and <strong>b </strong>are parameters that are trainable. Several such mappings, or <em>layers</em>, are concatenated to form a full network.

<strong>Computing the gradients. </strong>To derive the equations for backpropagation we want to express <em>, </em>as an expression containing the gradient <em><u><sup>∂L</sup></u><sub>∂</sub></em><strong><sub>z </sub></strong>with respect to the pre-activation value <strong>z</strong>. Using the chain rule we get

<em>.                                                      </em>(2)

We also need to compute the gradient with respect to the parameters <strong>W </strong>and <strong>b</strong>. To do this, again use the chain rule,

(3)

(4)

<strong>Exercise 1 (10 points): </strong>Derive expressions for <em><u><sup>∂L</sup></u><sub>∂</sub></em><strong><sub>x</sub></strong><em>, <sub>∂</sub><u><sup>∂L</sup></u></em><strong><sub>W </sub></strong>and <em><u><sup>∂L</sup></u><sub>∂</sub></em><strong><sub>b </sub></strong>in terms of <em><u><sup>∂L</sup></u><sub>∂</sub></em><strong><sub>z</sub></strong><em>,</em><strong>W </strong>and <strong>x</strong>. Include a <em>full derivation </em>of your results. The answers should all be given as matrix expressions without any explicit sums.

<strong>Checking the gradients. </strong>When you have code that uses gradients, it is very important to check that the gradients are correct. A bad way to conclude that the gradient is correct is to manually look at the code and convince yourself that it is correct or just to see if the function seems to decrease when you run optimization; it might still be a descent direction even if it is not the gradient. A much better way is to check finite differences using the formula

<em>,                                        </em>(5)

where <strong>e</strong><em><sub>i </sub></em>is a vector that is all zero except for position <em>i </em>where it is 1, and <em> </em>is a small number. In the code there is a file tests/test_fully_connected.m where you can test your implementation.

<strong>Computing the gradients of batches. </strong>When you are training a neural network it is common to evaluate the network and compute gradients not with respect to just one element but <em>N </em>elements in a batch. We use superscripts to denote the elements in the batch. For the dense layer above, we have multiple inputs <strong>x</strong><sup>(1)</sup>, <strong>x</strong><sup>(2)</sup>, <em>…</em>, <strong>x</strong><sup>(<em>N</em>) </sup>and we wish to compute <strong>z</strong>(1) = <strong>Wx</strong>(1) + <strong>b</strong>, <strong>z</strong>(2) = <strong>Wx</strong>(2) + <strong>b</strong>, ···, <strong>z</strong>(<em>N</em>) = <strong>Wx</strong>(<em>N</em>) + <strong>b</strong>. In the code for the forward pass you can see that the input array first is reshaped to a matrix <strong>X </strong>where each column contains all values for a single batch, that is,

<strong>X </strong>= <strong>x</strong><sup>(1)             </sup><strong>x</strong><em>.                                           </em>(6)

For instance, if <strong>x</strong><sup>(1) </sup>an image of size 5×5×3 it is reshaped to a long vector with length 75. We wish to compute

<strong>Z </strong>= <strong>z</strong>(1)                                                    <strong>z</strong><strong> Wx</strong>(1) + <strong>b Wx</strong>(2) + <strong>b </strong><em>… </em><strong>Wx</strong>

When we are backpropagating to <strong>X </strong>we have to compute

(8)

using

<em> .                                          </em>(9)

Use your expression obtained in Exercise 1 and simplify to matrix operations. For the parameters, we have that both <strong>W </strong>and <strong>b </strong>influence all elements <strong>z</strong><sup>(<em>i</em>)</sup>, so for the parameters we compute  using the chain rule with respect to each element in each <strong>z</strong><sup>(<em>i</em>) </sup>and just sum them up. More formally,

(10)

<em>.                                          </em>(11)

Note that the inner sum is the same as you have computed in the previous exercise, so you just sum the expression you obtained in the previous exercise over all elements in the batch. You can implement the forward and backward passes with for-loops, but it is also possible to use matrix operations to vectorize the code and for full credit you should vectorize it. Potentially useful functions in Matlab include bsxfun, sub2ind, ind2sub, reshape and repmat.

Using for-loops in programming can slow down your programm significantly. This is because in a for-loop, the data is being processed sequentially and not in parallel. Our aim is to parallelise the tasks in backpropagation by vectorising our Matlab commands and to avoid a for-loop running over all the data points.

<strong>Exercise 2 (10 points): </strong>Derive expressions for <strong>Z</strong>, <em><sub>∂</sub><u><sup>∂L</sup></u></em><strong><sub>X</sub></strong>, <em><sub>∂</sub><u><sup>∂L</sup></u></em><strong><sub>W </sub></strong>and

<em><u><sup>∂L</sup></u></em><em>∂</em><strong>b        </strong><em><u><sup>                        </sup></u></em>in terms of<sup>         </sup><em><sub>∂</sub><u><sup>∂L</sup></u></em><strong><sub>Z</sub></strong><em>,</em><strong>W</strong><em>,</em><strong>X </strong>and <strong>b</strong>. Include a <em>full derivation </em>of your results. Also add Matlab code that implements these vectorised expressions that you have just derived. In the code there are two files layers/fully_connected_forward.m and layers/fully_ connected_backward.m. Implement these functions and check that your implementation is correct by running tests/test_fully_ connected.m. For full credit your code should be vectorized over the batch. Include the relevant code in the report.

<h2>1.2         ReLU</h2>

The most commonly used function as a nonlinearity is the rectified linear unit (ReLU). It is defined by

ReLU : R → R;        <em>x<sub>i </sub></em>→7    <em>z<sub>i </sub></em>:= max(<em>x<sub>i</sub>,</em>0)<em>.                               </em>(12)

<table width="444">

 <tbody>

  <tr>

   <td width="444"><strong>Exercise 3 (10 points): </strong>Derive the backpropagation expression forin terms of . Give a full solution. Implement the layer in layers/relu_forward.m and layers/relu_backward.m. Test it with tests/test_relu.m. For full credit you must use built in indexing and not use any for-loops. Include the relevant code in the report.</td>

  </tr>

 </tbody>

</table>

<h2>1.3         Softmax Loss</h2>

Suppose we have a vector <strong>x </strong>= [<em>x</em><sub>1</sub><em>,…,x<sub>m</sub></em>]<sup>&gt; </sup>∈ R<em><sup>m</sup></em>. The goal is to classify the input as one out of <em>m </em>classes and <em>x<sub>i </sub></em>is a score for class <em>i </em>and a larger <em>x<sub>i </sub></em>is a higher score. By using the softmax function we can interpret the scores <em>x<sub>i </sub></em>as probabilities. Let

(13)

be the probability for class <em>i</em>. Now suppose that the ground truth class is <em>c</em>. Since now <em>z<sub>i </sub></em>are probabilities we can define the loss <em>L </em>to be minimized as the negative log likelihood,

<table width="444">

 <tbody>

  <tr>

   <td width="444"><strong>Exercise 4 (10 points): </strong>Compute the expression for  in terms of <em>z<sub>i</sub></em>. Write a full solution. Note that this is the final layer, so there are no gradients to backpropagate. Implement the layer in layers/softmaxloss_forward.m and layers/ softmaxloss_backward.m. Test it with tests/test_softmaxloss.m. Make sure that tests/test_gradient_whole_net.m runs correctly when you have implemented all layers. For full credit you should use built in functions for summation and indexing and not use any for-loops. Include the relevant code in the report.</td>

  </tr>

 </tbody>

</table>

<h1>2           Training a Neural Network</h1>

The function we are trying to minimize when we are training a neural net is

)                                        (15)

where <strong>w </strong>are all the parameters of the network, <strong>x</strong><sup>(<em>i</em>) </sup>is the input and <em>y</em><sup>(<em>i</em>) </sup>the corresponding ground truth and <em>L</em>(<strong>x</strong><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>;<strong>w</strong>) is the loss for a single example, given for instance by a neural net with a softmax loss in the last layer.In practice, <em>N </em>is very large and we never evaluate the gradient over the entire sum. Instead we evaluate it over a batch with elements because the network can be trained much faster if you use batches and update the parameters in every step.

If you are using gradient descent to train the network, the way the parameters are updated is

<strong>w</strong>                                             (16)

where <em>α </em>is a hyperparameter called the learning rate. Since we only evaluate the gradient over a few examples, the estimated gradient in (16) might be very noisy. The idea behind gradient descent with momentum is to average the gradient estimations over time and use the smoothed gradient to update the parameters. The update in (16) is modified as

<em>∂L</em>

<strong>m</strong><em><sub>n </sub></em>= <em>µ</em><strong>m</strong><em><sub>n</sub></em><sub>−1 </sub>+ (1 − <em>µ</em>)                                            (17)

<em>∂</em><strong>w</strong>

<strong>w</strong><em><sub>n</sub></em><sub>+1 </sub>= <strong>w</strong><em><sub>n </sub></em>− <em>α</em><strong>m</strong><em><sub>n                                                                                                 </sub></em>(18)

where <strong>m</strong><em><sub>n </sub></em>is a moving average of the gradient estimations and <em>µ </em>is a hyperparameter in the range 0 <em>&lt; µ &lt; </em>1 controlling the smoothness.

<strong>Exercise 5 (10 points): </strong>Implement gradient descent with momentum in training.m. Remember to include weight decay. Include the relevant code in the report.

<h1>3           Classifying Handwritten Digits</h1>

<strong>You must have solved all previous problems before you can start working on this and the next problem.</strong>

In mnist_starter.m there is a simple baseline for the MNIST dataset. Read the comments in the code carefully. It reaches about 98 % accuracy on the test set (this of course varies a bit from time to time). Validate this to make sure that the code you have written so far is correct.

<strong>Exercise 6 (25 points): </strong>Plot the filters the first convolutional layer learns. Plot a few images that are misclassified. Plot the confusion matrix for the predictions on the test set and compute the precision and the recall for all digits. Write down the number of parameters for all layers in the network. Write comments about all plots and figures.

<h1>4           Classifying Tiny Images</h1>

In cifar10_starter.m there is a simple network that can be trained on the CIFAR10 dataset. This baseline give an accuracy of about 48 % after training for 5000 iterations. Note that it is much harder to get good classification results on this dataset than MNIST, mainly due to significant intraclass variation.

<strong>Exercise 7 (25 points): </strong>Do whatever you want to improve the accuracy of the baseline network. Write what you have tried in the report, even experiments that did not work out. For your final model, compute and report all plots and numbers that were required in the previous exercise and comment on it.