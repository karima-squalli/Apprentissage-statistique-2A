function [W] = vect_w(x,x_mean,U)
%W Summary of this function goes here
%   Detailed explanation goes here

l = length(U(1,:));
W = zeros(l,1);

for i=1:l
    W(i) = sum((x-x_mean).*U(:,i));
end

end

