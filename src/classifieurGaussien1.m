function [m] = classifieurGaussien1(data_w,moyenne_intra,cov,m,mbis)
    taille = size(data_w); 
    inter =  norm((cov^(-1/2))*(data_w - moyenne_intra(:,1)))^(2); 
    min = inter;
    index = 1; 
    for j =1:m
        inter = norm((cov^(-1/2))*(data_w - moyenne_intra(:,j)))^(2);
        if(min>inter)
            min = inter;
            index = j; 
        end
    end
    m=mbis(index); 
end