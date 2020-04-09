function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


s = size(X, 1);

b = ones(1, s);		% a vector of bias
%fprintf('\nSize of b: %i \n', size(b))
a1 = [b',  X];	% add bias unit to X to form a1

%fprintf('\nSize of a1: %i \n', size(a1))
%fprintf('\nSize of Theta1: %i \n', size(Theta1))
z2 = a1 * Theta1';		%fprintf('\nSize of z2: %i \n', size(z2))

a2 = sigmoid(z2);

%fprintf('\nSize of a2: %i \n', size(a2))

bias = ones(1, size(a2, 1));
a2 = [bias', a2];	%fprintf('\n New Size of a2 : %i \n', size(a2))
%fprintf('\nSize of Theta2 : %i \n', size(Theta2))

z3 = a2 * Theta2';
a3 = sigmoid(z3);
%fprintf('\nSize of a3: %i \n', size(a3))
%fprintf('\nSize of z3: %i \n', size(z3))

y_matrix = eye(num_labels)(y,:);
% Compute cost
red = (- y_matrix) .* log(a3);
blue = (1 - y_matrix) .* log(1 - a3);


J = (1/m) * sum(sum(red - blue));

%theta1Sq = Theta1' * Theta1;
%theta2Sq = Theta2' * Theta2;

%fprintf('\nSize of Theta1 square: %i \n', size(theta1Sq))
%fprintf('\nSize of Theta2 square: %i \n', size(theta2Sq))

% Compute Regularization  and omit first rows of Theta1 and Theta2
iter = sum(sum(Theta1(:,2:end) .* Theta1(:,2:end))) + sum(sum(Theta2(:,2:end) .* Theta2(:,2:end)));
J =  J + (lambda/(2*m)) * iter;		% Add to Initial unregularized cost. This is cost loaded with ex4weights
% -------------------------------------------------------------


% =========================================================================

% Unroll gradients
d3 = a3 .- y_matrix;		
%fprintf('\nSize of d3: %i \n', size(d3))
z2 = a1 * Theta1';
%fprintf('\nSize of z2: %i \n', size(z2))
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
%fprintf('\nSize of d2: %i \n', size(d2))
%fprintf('\nSize of sigmoid z2: %i \n', size(sigmoid(z2)))
delta1 = d2' * a1;
%fprintf('\nSize of delta1: %i \n', size(delta1))
delta2 = d3' * a2;
%fprintf('\nSize of delta2: %i \n', size(delta2))
Theta1_grad = (1/m) * (delta1);
%fprintf('\nSize of Theta1_grad: %i \n', size(Theta1_grad))
Theta2_grad = (1/m) * (delta2);
%fprintf('\nSize of Theta2_grad: %i \n', size(Theta2_grad))

%Theta1
%Theta2

% Regulatization
Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1 = (lambda/m) .* (Theta1);
%fprintf('\nSize of Theta1: %i \n', size(Theta1))
Theta2 = (lambda/m) .* (Theta2);
%fprintf('\nSize of Theta2: %i \n', size(Theta2))

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
