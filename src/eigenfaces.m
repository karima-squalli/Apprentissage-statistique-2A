% P. Vallet (Bordeaux INP), 2019

clc;
clear;
close all;


%% Data extraction
% Training set
adr = '../database/training1/';
fld = dir(adr);
nb_elt = length(fld);

% Data matrix containing the training images in its columns 
data_trn = []; 

% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end

 
%%
% Size of the training set
[P,N] = size(data_trn);

% Classes contained in the training set
[~,I]=sort(lb_trn);
data_trn = data_trn(:,I);
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 

% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 

% Display the database
F = zeros(192*Nc,168*max(size_cls_trn));
for i=1:Nc
    for j=1:size_cls_trn(i)
          pos = sum(size_cls_trn(1:i-1))+j;
          F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
    end
end
figure;
imagesc(F);
colormap(gray);
axis off;


% % img contient la matrice 192 × 168 de pixels de l’image à l’adresse adr
% img = double(imread(adr)) ;
% 
% % imgv contient un vecteur de taille p = 32256
% imgv = img( :) ;
% 
% % colormap(gray) indique que l’affichage de l’image doit se faire en niveaux de gris
% imagesc(img) ;
% colormap(gray) ;

%% EIGENFACES


% QST 1


moyenne =  mean(data_trn,2);
X = 1/sqrt(N)*(data_trn - moyenne);

% vect propres et val propres associés à la matrice de Gram
[V, D] = eig(X'*X); 
V = flip(V(:,2:N),2);
U = X*V*((X*V)'*X*V)^(-1/2);
lambda = flip(diag(D(2:N,2:N)));
[l, ~] = size(lambda); 

%%%%% contribution energetique
energie = ones(l,1); 
for j=1:l
    for i=1:N
       energie(j) = energie(j)+ (sum(U(:,j).*(data_trn(:,i)-moyenne))^(2) )*1/N; 
    end
end 

figure, 
p=plot(linspace(1,59,59), energie,'b-o'); 
p.Color = 	'#77AC30';
hold on
axis([0 30 0 inf]);
xlabel("Index du vecteur propre (j)"); 
ylabel("Variance empirique");
title("Variance empirique suivant j")
%%
h = 192;
w = 168;
nl = 6;
nc = N/nl;
figure,
for i=1:nc*nl-1
    subplot(nl,nc,i);

    imagesc(reshape(U(:,i), [h,w]))
    hold on; 
        title("eigenface "+ num2str(i));
    colormap('gray');
    axis off
end
%%


l = 59;
n_elements = 6; 
 data_x = zeros(P, n_elements);
    
 

 
for i=1:n_elements
    data_x(:,i) = data_trn(:,(i-1)*10+1); 
end

data_reconstruite = zeros(P, n_elements); 

for j=1:n_elements

    for i=1:l
        data_reconstruite(:,j) = data_reconstruite(:,j) + (U(:,i)'*(data_x(:,j)-moyenne))*U(:,i);
    end
end

figure,
for i=1:n_elements
    subplot(2,n_elements,i+n_elements);
    imagesc(reshape(data_reconstruite(:,i)+moyenne, [h,w])); % c pas juste
    axis off; 
    hold on ;
        title("sujet "+num2str(i)+" après ACP");

    subplot(2,n_elements,i);
    imagesc(reshape(data_x(:,i), [h,w])); % c pas juste
    axis off;
    colormap('gray');
    axis off
    title("sujet "+num2str(i)+ " avant ACP");
end
%%
data_reconstruite(:,1) = data_reconstruite(:,1) + (U(:,i)'*(data_x(:,1)-moyenne))*U(:,i);
figure, 
imagesc(reshape(data_reconstruite(:,1), [h,w]));
colormap('gray')

% Remarque : jsp (:)(:):D

%%
% QST 4

s=0;
for i=1:N
    s = s + 1/N * norm(data_trn(:,i)-moyenne)^2;
end

k = zeros(1,l);
test = true;
alpha = 0.9;
lim = 0;
for i=1:l
    k(i) = sum(lambda(1:i))/s; 
    if (test) && (k(i)>=alpha)
        dim_opt = i;
        test = false;
    end
end
x = (1:l);


figure,
p1 = plot(x,k,'b-o');
p1.Color = 	'#77AC30';
p1.DisplayName = "Eigenface"
hold on 
line = xline(dim_opt);
line.DisplayName = "Limite à 90%";
titre = strcat('Ratio de reconstruction k(l) pour l=',num2str(l));
title(titre)
xlabel('l')
ylabel('k(l)')
legend();

% Commentaire : A partir de l = 8, les eigenfaces ne portent pas bcp d'info

% QST 5

% Si on utilise training2, on aura une dim optimale inferieure à celle dans 
% le training 1

%% CLASSIFIEUR k-NN

%% Data extraction
% Test set
adr = '../database/test2/';
fld = dir(adr);
nb_elt = length(fld);

% Data matrix containing the test images in its columns 
data_tst = []; 

% Vector containing the class of each test image
lb_tst = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_tst = [lb_tst ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_tst = [data_tst img(:)];
    end
end

% Size of the test set
[P,N_tst] = size(data_tst);

% Classes contained in the test set
[~,I]=sort(lb_tst);
data_tst = data_tst(:,I);
[cls_tst,bd,~] = unique(lb_tst);
Nc = length(cls_tst); 

% Number of test images in each class
size_cls_tst = [bd(2:Nc)-bd(1:Nc-1);N_tst-bd(Nc)+1]; 

% Display the database
F_tst = zeros(192*Nc,168*max(size_cls_tst));
for i=1:Nc
    for j=1:size_cls_tst(i)
          pos = sum(size_cls_tst(1:i-1))+j;
          F_tst(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_tst(:,pos),[192,168]);
    end
end
figure;
imagesc(F_tst);
colormap(gray);
axis off;


m = 6; % Nombre de sujets
m = 6; % Nombre de sujets
err_knn_b = zeros(1,30); 
for k=1:40
    C = zeros(N/m,m);
    for i=1:N/m
        for j=1:m
            C(i,j) = i + (j-1)*N/m;
        end
    end

    W = zeros(l,N);
    for i=1:N
        W(:,i) = vect_w(data_trn(:,i), moyenne,U);
    end


% VECTEUR X : test avc un visage 
    x = data_tst(:,1); % premier visage
    PHI = classifieurknn(x, moyenne, U, k, N, W, C, m,unique(lb_tst));


%%

    F_class = F(h*(PHI-1)+1:h*PHI,:);
    

% QST 2 CLASSIFIEUR

    x_pred = zeros(N_tst,1);

    for i=1:N_tst
        x_pred(i) = classifieurknn(data_tst(:,i), moyenne, U, k, N, W, C, m, unique(lb_tst));
    end
%%
    [mat_confusion_knn, err_knn] = confmat(lb_tst, x_pred);
    err_knn_b(k) = err_knn; 
end
F_class = F(h*(PHI-1)+1:h*PHI,:);

figure
    subplot(211)
    imagesc(reshape(x,[h,w]))
    colormap(gray)
    axis off
    title('Image de la data (test)')
    subplot(212)
    imagesc(F_class);
    colormap(gray);
    axis off
    title('Classe correspondante de la data (training) : C.kNN')

dim_opti = (1 -err_knn_b) < 0.90; 
[~,dim] = max(dim_opti); 
    figure, 
    p = plot(1:2:40,1-err_knn_b(1:2:end));
    p.Color = 'red'; 
    line = xline(dim);
    title('accuracy selon k'); 
    xlabel('k');
    ylabel('accuracy');


%%

F_class = F(h*(PHI-1)+1:h*PHI,:);
figure
subplot(211)
imagesc(reshape(x,[h,w]))
colormap(gray)
axis off
title('Image de la data (test)')
subplot(212)
imagesc(F_class);
colormap(gray);
axis off
title('Classe correspondante de la data (training) : C.kNN')

% QST 2 CLASSIFIEUR

x_pred = zeros(N_tst,1);

for i=1:N_tst
    x_pred(i) = classifieurknn(data_tst(:,i), moyenne, U, k, N, W, C, m, unique(lb_tst));
end
%%
[mat_confusion_knn, err_knn] = confmat(lb_tst, x_pred)

%% Classifieur gaussien
%%%% remarques: pour (2,3) changer U(:,i:j) pour U_util

%% Question 1

mbis = unique(lb_trn); 
[nb_sujets,~]= size(mbis) ; 
sujets = ones(h*w,nc,nb_sujets); 


% on stocke  les sujets dans une matrice de taille (h*w)xncxnb_sujets

for i=1:nb_sujets
   sujets(:,:,i) = data_trn(:,lb_trn ==mbis(i)); 
end

U_util = U(:,1:2); 
lbis = 2;
sortie = ones(lbis,nc, nb_sujets); 
for i=1:nb_sujets
    for d=1:nc
        sortie(:,d,i) = vect_w(sujets(:,d,i),moyenne,U_util);
    end
end

%%%%%% moyennes intra-classes mu = (1/nc)*(sum(w(x_i))
moyennes_intra = ones(2, nb_sujets);
for i=1:nb_sujets
    moyennes_intra(:,i) = (sum(sortie(:,:,i),2))/(nc);
end

str = ['r','g','y','m','c','b']; 

figure,
for i=1:nb_sujets
    scatter(sortie(1,:,i),sortie(2,:,i), str(i)); 
    hold on; 
    scatter(moyennes_intra(1,i), moyennes_intra(2,i), str(i), "filled");
    hold on; 
    legend();     
    
end
xlabel("Première composante"); 
ylabel("Deuxième composante");
title("Représentation des nuages de points (1,2)"); 

%% Classifieur gaussien V2

%%%%%%%%%%%définition des variables
mbis = unique(lb_tst); 
[nb_sujets,~] = size(mbis); 
[~, bd_taille] = size(data_tst);
nc = bd_taille/nb_sujets; 
sujets = ones(h*w,nc,nb_sujets); 
for i=1:nb_sujets
   sujets(:,:,i) = data_tst(:,lb_tst == mbis(i)); 
end
%%%%%%%%%%%calcul de w(x_i) sur toute la data_test
sortie = ones(l,nc, nb_sujets); 
for i=1:nb_sujets
    for d=1:nc
        sortie(:,d,i) = vect_w(sujets(:,d,i),moyenne,U);
    end
end

moyennes_intra = ones(l, nb_sujets);
for i=1:nb_sujets
    moyennes_intra(:,i) = (sum(sortie(:,:,i),2))/(nc);
end

sigma = mat_cov(sortie, moyennes_intra,h*w);

%%%%%%%%%%%% essaie sur data_tst(:,1)
tst = vect_w(data_tst(:,26),moyenne,U);
classe = classifieurGaussien1(tst,moyennes_intra,sigma,m,mbis); 

%%

%%%%%%%%%%%% test accuracy classifieur gaussien , regarder si MSE théorique
%%%%%%%%%%%% == MSE pratique, calcul du biais et de la variance

taille = size(data_tst); 
estim = ones(taille(2),1); 
for ii=1:taille(2)
    lbl = lb_tst(ii); 
    classe = classifieurGaussien1(vect_w(data_tst(:,ii),moyenne,U), moyennes_intra,sigma,m,mbis); 
    estim(ii) = mean((classe - lbl)^(2));
end
MSE = mean(estim); 

%% Amélioration classifieur guaussien 

%%%%%%% On calcule la matrice de covariance intra-classe 

%%% 
Sigma = zeros(l,l,nb_sujets); 
for i=1:nb_sujets
    Sigma(:,:,i) = mat_cov_intra(sortie(:,:,i), moyennes_intra(:,i),h*w); 
end

esti = classifieurGaussien2(tst,moyennes_intra, Sigma, nb_sujets,mbis);



%%

% VECTEUR X : test avc un visage 
x = data_tst(:,13); 


[class, phi] = classifieurGaussien(C,U, W, moyenne, x, m,l,unique(lb_tst));

% mat conf
x_pred = zeros(N_tst,1);
phii = zeros(N_tst,1);

for i=1:N_tst
    [x_pred(i), phii(i)] = classifieurGaussien(C, U, W, moyenne, data_tst(:,i), m, l, unique(lb_tst));
end
%%

[mat_confusion_gauss, err_gauss] = confmat(lb_tst, x_pred)


%% Figures

F_class2 = F(h*(phi-1)+1:h*phi,:);
figure
subplot(211)
imagesc(reshape(x,[h,w]))
colormap(gray)
axis off
title('Image de la data (test)')
subplot(212)
imagesc(F_class2);
colormap(gray);
axis off
title('Classe correspondante de la data (training g) : C.gaussien')
