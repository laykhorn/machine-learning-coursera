function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


	h = X * theta;		% Compute hypothesis
    err_vec = h - y;	% difference of hypothesis and result
    p = sum(err_vec .^ 2) ;
    u = (1/(2*m)) .* p;		% Unregularised cost

    % Calculate vectorised gradient
    u_grad = (1/m) * (X' * err_vec);
	grad = u_grad;


    % Regulatisation
	theta(1) = 0;				% Set theta(1) to 0
	thetasquare =  sum(theta.^2);

	r = (lambda / (2 * m)) .* thetasquare;

	r_grad  = theta .* (lambda / m);			
	
	% Return all values
	J = u + r;
	grad = u_grad + r_grad;




% =========================================================================

grad = grad(:);

end
