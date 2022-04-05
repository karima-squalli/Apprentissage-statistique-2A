function [sigma] = mat_cov(vects_w, moyennes_intra,n)
    taille = size(vects_w);    
    somme = zeros(taille(1),taille(1)); 
    for m=1:taille(3)
        for i=1:taille(2)
            somme = somme + ((vects_w(:,i,m) - moyennes_intra(:,m))*(vects_w(:,i,m) -moyennes_intra(:,m))'); 
        end
    end
    sigma = somme/n; 
end