function [classe] = classifieurknn(x, x_mean, U, k, N, W, C, m, lb)
%
%

w_x = vect_w(x, x_mean, U);



% Calcul de V
V_i = zeros(N,1);
for i=1:N
    V_i(i) = norm(w_x - W(:,i));
end
[~, V] = mink(V_i,k);

%
PHI_i = zeros(m,1);

for j=1:length(V)
    for i=1:m
        if (any(C(:,i) == V(j)))
            PHI_i(i) = PHI_i(i)+1;
        end
    end
end
[~, PHI] = max(PHI_i); 
classe = lb(PHI);

end

