function invA = tinv_NN_multi(A,lambdas,alpha)
% Find the inverse of a (covariance) tensor by using tensor nuclear
% norm (TNN) for multiple lambdas
% TNN is applied by using t-SVD
% Here we weighted each lambda according to value of the frequencies so that 
% high frequencies are penalized more and low frequencies penalized less 
% (sort of smoothness constraint)
% INPUTS
% A      : Third order tensor
% lambdas: A vector for the regularization parameters of TNN
% alpha  : Mixture constant
% 
% OUTPUTS
% invA  : inverses of the A found by applying TNN on t-SVD calculated for
% each lambda
%
%
% Copyright (C) 2015 Esin Karahan*, Pedro Ariel Rojas-Lopez', Pedro A. Valdes-Sosa'.
% * Bogazici University, Istanbul, Turkey, ' Cuban Neuroscience Center, Havana, Cuba
% contact: esin.karahan@gmail.com, pedro.rojas@cneuro.com, peter@cneuro.com
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Version 1 - May 2015

[l,m,n]=size(A);
Ahat = fft(A ,[],3);
Uhat = zeros(l,l,n);
Shat = zeros(l,m,n);
Vhat = zeros(m,m,n);
nn = max(l,m);

nlam = length(lambdas);
C    = zeros(m,l,n,nlam); 
invA = zeros(m,l,n,nlam); 
if ~mod(n,2) % if the length of the series is even
    half=n/2 + 1;
else % if the length of the series is odd
    half=ceil(n/2);
end

for k=1:n
    [Uhat(:,:,k),Shat(:,:,k),Vhat(:,:,k)]=svd(Ahat(:,:,k));
    
    % nuclear norm on S
    di = diag(Shat(:,:,k));
    di = di(di~=0);
    di = di.^2; % eigenvalue
    for ilam=1:nlam
        lambdai = lambdas(ilam);
        % weight lambda with frequency
        if k <= half  % positive frequency
            lambda  = lambdai*2^(k-1); %starts with DC 
        else % negative frequency
            lambda  = lambdai*2^(k - half );  
        end
        % shrink the eigenvalues
        ei = (-nn + sqrt(nn^2+(4*lambda*alpha).*(nn.*di + lambda*(1-alpha))))./(2*lambda*alpha);
        ei(ei<0)=0;
        ei = flipud(1./sqrt(ei));%back to singular value
        ei(ei==Inf) = 0;
        Shat(:,:,k) = full(spdiags(ei,0,Shat(:,:,k)));
        % inverse
        C(:,:,k,ilam) = Vhat(:,:,k)*Shat(:,:,k)'*Uhat(:,:,k)';
    end
end

for ilam=1:nlam
    invA(:,:,:,ilam)=ifft(C(:,:,:,ilam),[],3);
end
