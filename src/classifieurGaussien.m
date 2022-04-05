function [class, phi] = classifieurGaussien(C,U, W, x_mean, x, m,l,lb)
%renvoie les moyennes et matrice de covariance estimées
%W represente une matrice dont chaque colonne correspond aux cp d'une image
%i d'un sujet j(60 colonnes et l lignes)
%x l'image à traiter


w_x = vect_w(x, x_mean, U);

esp = zeros(l,m);


for jj=1:m
    for ii=C(1,jj):C(1,jj)+9  %9=N/m-1
        esp(:,jj) = esp(:,jj)+W(:,ii);
    end
    esp(:,jj) = 1/norm(C(:,jj))*esp(jj);
end

cov = zeros(l,l);

for jj=1:m
    for ii=C(1,jj):C(1,jj)+9
        cov = cov + (W(:,ii)-esp(:,jj))*(W(:,ii)-esp(:,jj))';
    end
end

phi_i = zeros(m,1);        

for jj=1:m
    phi_i(jj) = norm(cov^(-1/2)*(w_x - esp(:,jj)));
end

[~,phi] = min(phi_i);
class = lb(phi);
end

