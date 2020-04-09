function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


	i = X * theta; % Compute hypothesis
	h = sigmoid(i);
	% h = (1 ./ (1 .+ exp (-i)));			% Compute Sigmoid
	
	red = (-y') * log(h);
	blue = (1-y)' * (log(1-h));
	w = red - blue;
	u =  w .* (1/m);		% Unregularised cost
 
	theta(1) = 0;					% Set theta(1) to 0
	thetasquare =  theta' * theta;

	% Regulatisation
	r = (lambda ./ (2 .* m)) .* sum(thetasquare);            % Compute Regularised Cost
	J = u .+ r;            % Unregularised + regularized

	p = h-y;
	unreg_grad = X' * p;
	unreg_grad = unreg_grad .* (1/m); 	% Calculate vectorised gradient
	reg_grad = theta .* (lambda/m);	% Regularised gradient

	grad = unreg_grad  .+ reg_grad;				% Gradient

% =============================================================

grad = grad(:);

end
