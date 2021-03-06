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



# Add a column of ones to X
X_padded = [ones(m,1), X];

Delta2 = zeros(num_labels, hidden_layer_size + 1);
Delta1 = zeros(hidden_layer_size, input_layer_size + 1);
J = 0;
for i = 1:m
	#
	# FORWARD PROPAGATION
	#
	a1 = X(i,:);
    a1_padded = [1, a1];
	z2 = a1_padded*Theta1';
	a2 = sigmoid(z2);
    a2_padded = [1, a2];
    z3 = a2_padded * Theta2';
	a3 = sigmoid(z3);

	# transform y into a vector
	y_vec = zeros(num_labels,1);
	y_vec(y(i)) = 1;
	
	J += sum( -y_vec .* log(a3') - (1 - y_vec) .* log( 1 - a3'));

	delta3 = a3' - y_vec;
	delta2 = (Theta2' * delta3) .* (a2_padded' .* (1 - a2_padded')); # sigmoidGradient(z2'); ??? it should be sigmoidGradtient of z2 with a bias?
	Delta2 = Delta2 + delta3 * a2_padded;
	Delta1 = Delta1 + delta2(2:end) * a1_padded;

end
J /= m;
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

# Regularization
reg = 0;
for j = 1:hidden_layer_size
	for k = 1:input_layer_size
		reg += Theta1(j, k + 1)^2; # Skip bias
	end
end
for j = 1:num_labels
	for k = 1:hidden_layer_size
		reg += Theta2(j, k + 1)^2; # Skip bias
	end
end

J += lambda/(2*m) * reg;


# Regularize Theta gradients
Reg1 = Theta1;
Reg1(:,1) = zeros(size(Theta1,1),1);
Theta1_grad += (Reg1 .* lambda/m);

Reg2 = Theta2;
Reg2(:,1) = zeros(size(Theta2,1),1)	;
Theta2_grad += (Reg2 .* lambda/m);

for j = 1:hidden_layer_size
	for k = 1:input_layer_size
		reg += Theta1(j, k + 1)^2; # Skip bias
	end
end
for j = 1:num_labels
	for k = 1:hidden_layer_size
		reg += Theta2(j, k + 1)^2; # Skip bias
	end
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
