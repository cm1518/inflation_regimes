function [pred,uncon_var,lmean] = predictability_katerina_fast( BETA,SIGMA,n,l,horz )


Noregimes = size(BETA,3);
pred = zeros(n, Noregimes,horz); %pred for all variables in all reg
uncon_var = zeros(n, n, Noregimes); %unconditional variance for all
lmean = zeros(n, Noregimes);

for j=1:Noregimes


 % Compute COGLEY, SARGENT PRIMICERI MEASURE, measure No 1 in the note
        OMEGA=sparse(zeros(n*l,n*l));
        OMEGA(1:n,1:n)=squeeze(SIGMA(:,:,j));  %companion variance
        [Ficom,mu]=companion(BETA(:,:,j)',n,l,1);
    
        Ficom_sp = sparse(Ficom) ;
        
        vecuvar = (speye((n*l)^2)-kron(Ficom_sp,Ficom_sp)) \ OMEGA(:) ;
        % vecuvar = (((eye((N*VarLag)^2)-kron(Ficom,Ficom))^(-1))*OMEGA(:)); %gamma0 the unconditional variance 
        
        uvar = full(reshape(vecuvar, n*l, n*l));
        % this the analytic expression here is the original function from Cogley et al AEJ Macro: uvar=doublej(FF,OMEGA);
    
       uncon_var (:,:,j) = uvar(1:n, 1:n) ;  %store the unconditional variance over t,useful to compute unconditional pairwise correlations for example
       lmeanALL = ((eye(n*l)-Ficom)^(-1))*[mu;zeros(n*(l-1),1)];
       lmean(:,j) = lmeanALL(1:n);

        for hh=1:horz
            FF=((Ficom))^hh;
            R2 = diag(FF*uvar*FF')./diag(uvar);
            pred(:,j,hh) = (R2(1:n));   
            % this is Cogley et al R2 measure; 
            % Katerina wrote this equivalent compuatation the original AEJ macro paper does this:
            % cvar=doublejH(FF,OMEGA,hh);
            % R2=1-(cvar(1:n,1:n)./uvar(1:n,1:n));
        end
end


end
        