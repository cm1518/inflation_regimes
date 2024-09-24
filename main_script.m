%%% Written by Katerina Jan 2024
%%% The code doee the following:
% 1. Loads data and put into VAR format
% 2. Construct the grid for the threshold 
% 3. Evaluates the posterior over the grid and numerically maximises
% 4. Given the estimated threshold, runs a Self-Exciting Bayesian VAR with NW priors
% 5. Generates some IRFs for monetary policy shock, using Romer and Romer
% 6. Computes persistence measures from Cogley at al 2010, as well as
% unconditional means and covariances in each regime
% shock
 
clear
close all
clc
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                             LOAD THE DATA                                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load DATAmonthly.mat
load RRMonshock.mat

%restricted sample due to limited availability of R&R proxy
DATA = DATAmonthly(1:456,:);

date = DATA(:,1);
% order 2 - ffr, 3 unmpl, 4 - infl

model.Y = [RRMonshock(13:end), DATA(:,2), DATA(:,3), DATA(:,4)]'; 

[N,~] = size(model.Y);   % number of vars
model.N = N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        %%%OPTIONS FOR THE VAR coefficients                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model.VarLag = 12; %lag order of the VAR
[ model.VarX,model.VarY,model.VarT ] = getdatavar( model.Y',model.VarLag );
model.T = model.VarT;
model.VarOverall_shrinkage = 1; %overall shrinkage for the priors of VAR parameters (this is lambda for Minnesota prior 
[model.VarPriorMean, model.VarPriorVar ] = get_Minnesota_prior( model.VarLag, model.VarOverall_shrinkage, model.VarY, 0);

%FIT A PRELIMINARY LINEAR VAR TO GET RESIDUALS
model.VarBB0=((model.VarX'*model.VarX)^(-1))*(model.VarX'*model.VarY);
resid=(model.VarY-model.VarX*model.VarBB0);
sigma0=resid'*resid/model.T;
model.Resid=resid';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%%OPTIONS FOR THE VOL parameters                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[model.VarVolPriorA, model.VarVolPriorB ] = get_priorsIW( model.VarLag, model.VarY );
%these IW prior for the volatility matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        %%%OPTIONS FOR THE STATE                                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K=3;  %select number of regimes  
d = 1; %lag of the state variable
model.state = DATA(model.VarLag+1-d:end-d,4);  %inflation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%% GRID CONSTRUCTION                                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grid construction
Ngridpoints = 500;
end_space = 0;
begin_space = 0;
space = 0; 
gam_values = linspace(min(model.state)+begin_space, max(model.state)-end_space, Ngridpoints);

posterior = NaN(Ngridpoints,Ngridpoints);
likelihood = NaN(Ngridpoints,Ngridpoints);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%% OBJECTIVE FUNCTION EVALUATION                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the code currently runs with 3 regimes
% to allow more regimes, add extra loops for the grid evaluation
% to reduce the number of regimes, delete the inside loops 

for i = 1:Ngridpoints

        gamma = gam_values(i);
%        posterior(i) = eval_objfcn_katerina(gamma, model.VarX, model.VarY, model.state, 1, model,0);

%%%%2 regimes
for j =i+space:Ngridpoints
         gamma = [gam_values(i), gam_values(j)]';
         posterior(i,j) = eval_objfcn_katerina(gamma, model.VarX, model.VarY, model.state, 1, model,1);
end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%% OBJECTIVE FUNCTION MAXIMISATION                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[postval, indexP] = max(posterior, [], 'all'); %objfcn at post
[rowP0, colP0] = ind2sub([Ngridpoints,Ngridpoints], indexP);

threshPO(1) = gam_values(rowP0); 
threshPO(2) = gam_values(colP0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%% PLOT OBJECTIVE FUNCTION                                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure 
h1 = surf(gam_values, gam_values, posterior)
% hold on
% h2 = surf(gam_values, gam_values, real(lik))
xlabel('gamma 1 grid')
xlim([min(model.state), max(model.state)])
ylabel('gamma 2 grid')
ylim([min(model.state), max(model.state)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               %%% SET-VAR ESTIMATION                                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R = K;         %No of regimes
ir_hor  = 96;  %IRF horizon

VarX = model.VarX;
VarY = model.VarY;
p = model.VarLag;
Nodraws = 5000; %No of posterior draws 
model.StabilityThreshold = 0.9999; %upper bound on max abs eigenvalue
model.VarCheckStability=1;         % 1 to check stability
 
% define first regime and last regime given the estimated threshold value

     % Define weights
     w(:,1) = (model.state<=threshPO(1)); 
     % Define weights
     w(:,R) = (model.state>threshPO(R-1));

% all other regimes
for rr=2:R-1
    %Define regime-dependent X  
     w(:,rr) = (model.state>threshPO(rr-1)).*(model.state<=threshPO(rr));
end

        %all priors are common across regimes
    priorprec = model.VarPriorVar^(-1);
    priormean  = model.VarPriorMean;
    priorB     = model.VarVolPriorB;
    priorA     = model.VarVolPriorA;
    g1         = priormean'*priorprec*priormean;

for rr=1:R 
    W               = diag(w(:,rr));
    Ni              = trace(W);
    postprec        = (priorprec+VarX'*W*VarX);
    post_v          = postprec^(-1);
    postpmean       = post_v*(VarX'*W*VarY+priorprec*priormean);
    postA           = priorA+Ni; % alpha0 + N_i
    g2              = VarY'*W*VarY;
    g3              = squeeze(postpmean)'*postprec*squeeze(postpmean);
    postB           = priorB+g1+g2-g3;
    postB           = 0.5*postB+0.5*postB';
    postpmode = vec(postpmean');
    %sigma posterior mode:

    %LOOP OVER POSTERIOR DRAWS
for it = 1:Nodraws
    check=0;
while check==0
Sigma = iwishrnd(squeeze(postB),postA); %draw a matrix sigma from IW
A0inv      = chol(Sigma,'lower');
nu=randn(N*p+1,N);
Bdraw=(squeeze(postpmean)+chol(squeeze(post_v))'*nu*(chol(squeeze(Sigma))));

if model.VarCheckStability==1
    [Ficom,~]=companion(Bdraw',N,p, 1);
if max(abs(eig(Ficom)))<model.StabilityThreshold %check if draw is stationary
     check=1; 
end
elseif model.VarCheckStability==0
    check=1;
end

end 

%store coefficients

samples.Bmat(it,:,:,rr) = Bdraw; 
samples.Sigma(it,:,:,rr) = Sigma;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%% STRUCTURAL ANALYSIS                                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Recursive identification - IRFs
ch=(chol(Sigma))';
d=diag(diag(ch));
A=ch*d^(-1); %this is lower triangular with ones on the main diagonal  
%NOTE that A*d*d'*A = Sigma 

b1 = A(:,1);
%normalize for 0.5 impact on fedfunds:
b1 = b1/(b1(2))*0.5;
irf (:,1)= b1;
irs0 = [b1; zeros((p-1)*N,1)];
        for j=1:ir_hor-1
            irs = (Ficom^j)*irs0;
            irf (:,j+1)= irs(1:N);
        end

samples.irs (:,:,it,rr) = irf;
end
end

samples.irs = sort (samples.irs,3);
IRF_med = squeeze(median(samples.irs ,3));

CI1_irf = squeeze(samples.irs(:,:,0.16*Nodraws,:));
CI2_irf = squeeze(samples.irs(:,:,0.84*Nodraws,:));

%%% IRF PLOTS

colour  = [0/255, 114/255, 189/255; 0/255, 191/255, 191/255; 119/255, 172/255, 48/255];

v = {'Federal Funds Rate', 'Unemployment', 'Inflation'};

figure
%% ALL REGIMES 
for jj=1:N-1
    for rr=1:R
        vv = jj+1; % start from variable 2, first variable is the shock

subplot(N-1,R,rr+R*(jj-1))
h1=plot(IRF_med(vv,:,rr),'Color', colour(:,rr), 'LineWidth', 1.2)
hold on
h4=plot(CI1_irf(vv,:,rr),'Color',  colour(:,rr), 'LineStyle', '--', 'LineWidth', 1.2)
 hold on 
h5 = plot(CI2_irf(vv,:,rr), 'Color',  colour(:,rr), 'LineStyle', '--', 'LineWidth', 1.2) 
hold on
plot(zeros(ir_hor-1,1), 'k', 'LineWidth', 1)

if jj==N-1
         legend('Median','68% CIs') 
end
title(v{jj})
            set(gca, 'FontName', 'Times New Roman', 'FontSize',11)
      grid on
      axis tight
     

    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%% REDUCED FORM ANALYSIS                                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    horizon        = 90;
    predictability = zeros (model.N, R, horizon, Nodraws);
    uncon_var      = zeros (model.N, model.N, R, Nodraws);
    lmean          = zeros (model.N, R, Nodraws);

for nsim=1:Nodraws
[predictability(:,:,:,nsim),uncon_var(:,:,:,nsim),lmean(:,:,nsim)] = predictability_katerina_fast(squeeze(samples.Bmat(nsim,:,:,:)),squeeze(samples.Sigma(nsim,:,:,:)),model.N,model.VarLag,horizon);
end

%predictability contains the persistence measure of the series, 
% dimension is NoVar x NoRegimes x NoHorizons x NoPostDraws

%uncon_var contains the unconditional covariance matrix of the series 
% dimension is NoVar x NoVar x NoRegimes x NoPostDraws

%lmean contains the unconditional LR means of the series 
% dimension is NoVar x NoRegimes x NoPostDraws
