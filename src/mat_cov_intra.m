function [sigma] = mat_cov_intra(data,moyenne,n)
    taille = size(data);    
    somme = zeros(taille(1),taille(1)); 
    for i=1:taille(2)
        somme = somme + ((data(:,i) - moyenne)*(data(:,i) -moyenne)'); 
    end
    sigma = somme; 
end

