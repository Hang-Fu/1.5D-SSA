%==========================================================================
% H. Fu, et al, "A Novel Spectral-Spatial Singular Spectrum Analysis Technique for Near
% Real-Time In Situ Feature Extraction in Hyperspectral Imaging"

% 1.5D-SSA method on Indian Pines dataset
%==========================================================================

close all;clear all;clc;
addpath(genpath('.\Dataset'));
addpath(genpath('.\libsvm-3.18'));
addpath(genpath('.\functions'));

%% data
load('indian_pines_gt'); img_gt=indian_pines_gt;
load('Indian_pines_corrected');img=indian_pines_corrected;
[Nx,Ny,bands]=size(img);

%% parameters
u=2; w=2*u+1; w2=w*w;  %neighborhood window size w
L=10;                  %the embedding window L
S=15;                  %the number of similar pixels S
tic;
%% 1.5D-SSA
for i=1:bands
    indian_pines(:,:,i) = padarray(img(:,:,i),[u,u],'symmetric','both');
end
img_SSA=zeros(Nx,Ny,bands);
for i=1:Nx
    for j=1:Ny
        i1=i+u;j1=j+u;
        testcube=indian_pines(i1-u:i1+u,j1-u:j1+u,:);
        m=reshape(testcube,[w2,bands]);          
        center(1,:)=img(i,j,:);  
        ED=zeros(1,w2);
        for ii=1:w2
            ED(:,ii)=sqrt(sum(power((m(ii,:)-center),2)));  %Euclidean distance
        end
        [~,ind]=sort(ED); 
        sel_m=m(ind(1:S),:); 
        sel_m=sel_m(:);                                     %the extended spectral vector
        rec_m=SSA(L,1,1,sel_m');
        a=reshape(rec_m,[S,bands]);
        img_SSA(i,j,:)=a(1,:);
    end
end

%% training-test samples
Labels=img_gt(:);    
Vectors=reshape(img_SSA,Nx*Ny,bands);  
class_num=max(max(img_gt))-min(min(img_gt));
trainVectors=[];trainLabels=[];train_index=[];
testVectors=[];testLabels=[];test_index=[];
Samp_pro=0.1;                                                         %proportion of training samples
for k=1:1:class_num
    index=find(Labels==k);                  
    perclass_num=length(index);           
    Vectors_perclass=Vectors(index,:);    
    c=randperm(perclass_num);                                      
    select_train=Vectors_perclass(c(1:ceil(perclass_num*Samp_pro)),:);    %select training samples
    train_index_k=index(c(1:ceil(perclass_num*Samp_pro)));
    train_index=[train_index;train_index_k];
    select_test=Vectors_perclass(c(ceil(perclass_num*Samp_pro)+1:perclass_num),:); %select test samples
    test_index_k=index(c(ceil(perclass_num*Samp_pro)+1:perclass_num));
    test_index=[test_index;test_index_k];
    trainVectors=[trainVectors;select_train];                    
    trainLabels=[trainLabels;repmat(k,ceil(perclass_num*Samp_pro),1)];
    testVectors=[testVectors;select_test];                      
    testLabels=[testLabels;repmat(k,perclass_num-ceil(perclass_num*Samp_pro),1)];
end
[trainVectors,M,m] = scale_func(trainVectors);
[testVectors ] = scale_func(testVectors,M,m);   

%% SVM-based classification
Ccv=1000; Gcv=0.125;
cmd=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 
models=svmtrain(trainLabels,trainVectors,cmd);
testLabel_est= svmpredict(testLabels,testVectors, models);

%classification map
result_gt= Labels;       
for i = 1:1:length(testLabel_est)        
   result_gt(test_index(i)) = testLabel_est(i);  
end
result_map_l = reshape(result_gt,Nx,Ny);result_map=label2color(result_map_l,'india');figure,imshow(result_map);

%classification results
[OA,AA,kappa,CA]=confusion(testLabels,testLabel_est);
result=[CA*100;OA*100;AA*100;kappa*100];
toc;