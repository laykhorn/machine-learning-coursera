function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%


	i = X * theta; % Compute hypothesis
	h = (1 ./ (1 + exp (-i)));			% Compute Sigmoid

	red = -y .* log(h);
	blue = (1-y) .* (log(1-h));
	w = sum(red) - sum(blue);
	J = (1/m)*(w);		% Unregularised cost


% Regulatisation

theta(1) = 0;						% Set theta(1) to 0
thetasquare = sum(theta.^2);
r = (1 / (2 * m)) * thetasquare;


z = sum((h.-y) .* X);		% Calculate vectorised gradient
z = z / m;					% Scale by m
grad = transpose(z);		% Return gradient



% =============================================================

end
