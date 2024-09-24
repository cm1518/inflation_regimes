function objfcn = eval_objfcn_katerina(thresh, VarX, VarY, state , post, model , ssr)

% the function evaluates the log posterior at the posterior mode, just as in the paper
% thresh is the Rx1 vector of thresholds at which objfcn is evaluated
% VarX and VarY are the VAR X and Y variables; 
% state is the state variable
% model contains priors parameters from main code

R = size(thresh,1)+1; %number of regimes
k = size(VarY,2); %number of variables in VAR
N = size(VarY,1); %sample size
p = model.VarLag;

% first regime and last regime

     % Define weights
     w(:,1) = (state<=thresh(1)); 
     % Define weights
     w(:,R) = (state>thresh(R-1));

% all other regimes
for rr=2:R-1
    %Define regime-dependent X  
     w(:,rr) = (state>thresh(rr-1)).*(state<=thresh(rr));
end

% all priors are common across regimes
    priorprec = model.VarPriorVar^(-1);
    priormean  = model.VarPriorMean;
    priorB     = model.VarVolPriorB;
    priorA     = model.VarVolPriorA;
    g1         = priormean'*priorprec*priormean;

for rr=1:R 
    W               = diag(w(:,rr));
    Ni              = trace(W);

    postprec        = (priorprec+VarX'*W*VarX);
    postpmean       = (postprec^(-1))*(VarX'*W*VarY+priorprec*priormean);
    postA           = priorA+Ni; % alpha0 + N_i
    g2              = VarY'*W*VarY;
    g3              = squeeze(postpmean)'*postprec*squeeze(postpmean);
    postB           = priorB+g1+g2-g3;
    postB           = 0.5*postB+0.5*postB';

    postpmode = vec(postpmean');
    %sigma posterior mode:
    post_sigma_mode = (postA + k + 1)^(-1)*(postB); 
    priorvar   = kron(post_sigma_mode,priorprec^(-1));

    %evaluate prior for beta at posterior mode
    prior_b = log(mvnpdf(postpmode, priormean(:), priorvar)); 
    if prior_b == -Inf
       prior_b = -1000;
    end
    %evaluate prior for sigma at posterior mode
    prior_s = 0.5*priorA*log(det(priorB))-0.5*priorA*k*log(2);
    prior_s = prior_s - 0.5*(priorA+k+1)*log(det(post_sigma_mode));
    prior_s = prior_s - 0.5*trace(priorB*post_sigma_mode^(-1));
    prior (rr)= (Ni/N)*(prior_b+prior_s);

    YY = W*VarY;
    YYmean = W*VarX*postpmean;
    selperiods = find(w(:,rr))'; %periods in regime rr

for nn = [selperiods]  
try
   lik (rr,nn) = log(mvnpdf(YY(nn,:)',YYmean(nn,:)',post_sigma_mode));
catch 
   lik (rr,nn) = NaN;
end
end

end 
objfcn = sum(sum(lik))+sum(prior);
end 