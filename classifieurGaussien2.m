function [m] = classifieurGaussien2(data_w,moyennes_intra, Sigma, m,mbis)
%CLASSIFIEURGUASSIEN2 Summary of this function goes here
%   this function predict the class of a data with the gaussian classifier,
%   using the intra-class covariance matrixes
    inter = ones(m, 1); 
    for j =1:m
        inter(j) = norm((Sigma(:,:,j)^(-1/2))*(data_w - moyennes_intra(:,j)))^(2);
    end
    [x index] = min(inter); 
     m = mbis(index);  
end

