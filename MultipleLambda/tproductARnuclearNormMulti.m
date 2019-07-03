% This function is the caller function for Granger Causality analysis with
% t-product
%
% References:
% o Kilmer, M. E., Braman, K., Hao, N., & Hoover, R. C. (2013). Third-Order
% Tensors as Operators on Matrices: A Theoretical and Computational
% Framework with Applications in Imaging. SIAM Journal on Matrix Analysis
% and Applications, 34(1), 148–172. 
% o Dietrich, C. R., & Newsam, G. N. (1997). Fast and Exact Simulation of Stationary
% Gaussian Processes through Circulant Embedding of the Covariance Matrix.
% SIAM Journal on Scientific Computing. 
% o Chi, E. C., & Lange, K. (2014). Stable estimation of a covariance matrix 
% guided by nuclear norm penalties. Computational Statistics and Data Analysis, 80, 117–128.
% 
% Version 1 - May 2015
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the data X
load Y
% normalize the data to have zero mean and unit standard deviation
Y = zscore(Y);
Y = Y';
[Nv,Nt] = size(Y);
% Determine the model order of the AR
Nlag = 10;
p = (Nlag-1)*2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the first column of the autocovariance
C = calculateAutoCov(Y,Nlag);

% Circulant embedding of Autocovariance tensor
R = circEmbedAutoCov(C(:,:,1:Nlag));
r = circEmbedAutoCov(C(:,:,2:Nlag+1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set alpha according to Chi et. al.
% Eq. 5
trC=0;
for i=1:Nlag+1
    trC = trC+trace(C(:,:,i));
end
alpha=1/(1+(trC/(Nv*(Nlag+1))).^2);

% Find lambdamax
epsilon = 1e-2;
[~,~,~,~,Shat] = tusv(R);
di = zeros(Nv,p);
for k = 1:p
    di(:,k) = diag(Shat(:,:,k));
end
di = di(:);
t1 = abs(sqrt((1-alpha)/alpha)*Nt*di./(2*(1-alpha) - Nt/(2*alpha)));
t2 = epsilon*sqrt((1-alpha)/alpha);
lambdamax = max(t1./t2);

% test for various lambdas
% lambdas = logspace(1,log10(lambdamax),10);
% or just test for the maximum of the lambda
lambdas = max(lambdas);

nlam = length(lambdas);
Ac   = cell(nlam,1);

tic;
% Find the inverse of R through nuclear norm penalization for multiple
% lambdas
invR = tinv_NN_multi(R,lambdas,alpha);
% Calculate the AR coefficients for multiple lambdas
for i = 1:nlam
    A = fasttproduct(squeeze(real(invR(:,:,:,i))),r);
    A = A(:,:,1:Nlag);
    Ac{i} = A;
end
toc;

% Plot the results
for i=1:nlam
    figure
    imagesc(reshape((Ac{i}),[Nv,Nv*(Nlag)])),colorbar
    title(['weighted \lambda = ' num2str(lambdas(i)) ' x2^f'])
end

