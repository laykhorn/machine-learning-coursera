function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Clist = [0.01 0.03 0.1 0.3 1 3 10 30];
slist = [0.01 0.03 0.1 0.3 1 3 10 30];

result = zeros(length(Clist) * length(slist), 3);
row = 1;
%size(result)


for i=1:length(Clist)
	C = Clist(i);

	for j=1:length(slist)
		sigma = slist(j);
		mod = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		% size(mod), size(Xval), size(yval)
		predict = svmPredict(mod, Xval);
		err_val = mean(double(predict ~= yval));
		result(row,:) = [C  sigma err_val]; 
		row = row + 1;

	end
	
end

[m n] =  min(result(:,3))
C = result(n, 1);
sigma = result(n, 2);
err_val = result(n, 3);

% =========================================================================

end
