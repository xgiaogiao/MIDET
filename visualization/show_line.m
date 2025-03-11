%% plot color pics
clear; clc;
%load(['simulation_results\results\','truth','.mat']);

load(['simulation_results\results\','lambda_net','.mat']);
pred_block_hdnet = pred;
load(['simulation_results\results\','tsa_net','.mat']);
pred_block_mst_s = pred;

load(['simulation_results\results\','mst_l','.mat']);
pred_block_mst_m = pred;

load(['simulation_results\results\','cst_l','.mat']);
pred_block_mst_l = pred;

load(['simulation_results\results\','admm_net','.mat']);
pred_block_mst_plus_plus = pred;

load(['simulation_results\results\','dauhst_9stg','.mat']);
pred_block_tsanet = pred;

load(['simulation_results\results\','padut_12stg','.mat']);
pred_block_lambda_net = pred;

load(['simulation_results\results\','RCUMP','.mat']);
pred_block_RCUMP = hsi;

load(['simulation_results\results\','ours_9stg','.mat']);
pred_block_ours_9stg = pred;

lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0...
    516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0...
    614.5 625.0 636.5 648.0];

truth(find(truth>0.7))=0.7;
pred_block_hdnet(find(pred_block_hdnet>0.7))=0.7;
pred_block_mst_s(find(pred_block_mst_s>0.7))=0.7;
pred_block_mst_m(find(pred_block_mst_m>0.7))=0.7;
pred_block_mst_l(find(pred_block_mst_l>0.7))=0.7;
pred_block_mst_plus_plus(find(pred_block_mst_plus_plus>0.7))=0.7;
pred_block_tsanet(find(pred_block_tsanet>0.7))=0.7;
pred_block_lambda_net(find(pred_block_lambda_net>0.7))=0.7;
pred_block_RCUMP(find(pred_block_RCUMP>0.7))=0.7;
pred_block_ours_9stg(find(pred_block_ours_9stg>0.7))=0.7;

f = 10;

%% plot spectrum
figure(123);
[yx, rect2crop]=imcrop(sum(squeeze(truth(f, :, :, :)), 3));
rect2crop=round(rect2crop)
close(123);

figure; 

spec_mean_truth = mean(mean(squeeze(truth(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_hdnet = mean(mean(squeeze(pred_block_hdnet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst_s = mean(mean(squeeze(pred_block_mst_s(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst_m = mean(mean(squeeze(pred_block_mst_m(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst_l = mean(mean(squeeze(pred_block_mst_l(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst_plus_plus = mean(mean(squeeze(pred_block_mst_plus_plus(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
pred_block_tsanet = mean(mean(squeeze(pred_block_tsanet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
pred_block_lambda_net=mean(mean(squeeze(pred_block_lambda_net(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
pred_block_RCUMP=mean(mean(squeeze(pred_block_RCUMP(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
pred_block_ours_9stg=mean(mean(squeeze(pred_block_ours_9stg(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);



spec_mean_truth = spec_mean_truth./max(spec_mean_truth);
spec_mean_hdnet = spec_mean_hdnet./max(spec_mean_hdnet);
spec_mean_mst_s = spec_mean_mst_s./max(spec_mean_mst_s);
spec_mean_mst_m = spec_mean_mst_m./max(spec_mean_mst_m);
spec_mean_mst_l = spec_mean_mst_l./max(spec_mean_mst_l);
spec_mean_mst_plus_plus = spec_mean_mst_plus_plus./max(spec_mean_mst_plus_plus);
pred_block_tsanet = pred_block_tsanet./max(pred_block_tsanet);
pred_block_lambda_net = pred_block_lambda_net./max(pred_block_lambda_net);
pred_block_RCUMP=pred_block_RCUMP./max(pred_block_RCUMP);
pred_block_ours_9stg = pred_block_ours_9stg./max(pred_block_ours_9stg);


corr_hdnet = roundn(corr(spec_mean_truth(:),spec_mean_hdnet(:)),-4);
corr_mst_s = roundn(corr(spec_mean_truth(:),spec_mean_mst_s(:)),-4);
corr_mst_m = roundn(corr(spec_mean_truth(:),spec_mean_mst_m(:)),-4);
corr_mst_l = roundn(corr(spec_mean_truth(:),spec_mean_mst_l(:)),-4);
corr_mst_plus_plus = roundn(corr(spec_mean_truth(:),spec_mean_mst_plus_plus(:)),-4);
corr_tsanet = roundn(corr(spec_mean_truth(:),pred_block_tsanet(:)),-4);
corr_lambda_net = roundn(corr(spec_mean_truth(:),pred_block_lambda_net(:)),-4);
corr_RCUMP = roundn(corr(spec_mean_truth(:),pred_block_RCUMP(:)),-4);
corr_ours_9stg = roundn(corr(spec_mean_truth(:),pred_block_ours_9stg(:)),-4);

X = lam28;

Y(1,:) = spec_mean_truth(:); 
Y(2,:) = spec_mean_hdnet(:); Corr(1)=corr_hdnet;
Y(3,:) = spec_mean_mst_s(:); Corr(2)=corr_mst_s;
Y(4,:) = spec_mean_mst_m(:); Corr(3)=corr_mst_m;
Y(5,:) = spec_mean_mst_l(:); Corr(4)=corr_mst_l;
Y(6,:) = spec_mean_mst_plus_plus(:); Corr(5)=corr_mst_plus_plus;
Y(7,:) = pred_block_tsanet(:); Corr(6)=corr_tsanet;
Y(8,:) = pred_block_lambda_net(:); Corr(7)=corr_lambda_net;
Y(9,:) = pred_block_RCUMP(:); Corr(8)=corr_RCUMP;
Y(10,:) = pred_block_ours_9stg(:); Corr(9)=corr_ours_9stg;

createfigure(X,Y,Corr)


