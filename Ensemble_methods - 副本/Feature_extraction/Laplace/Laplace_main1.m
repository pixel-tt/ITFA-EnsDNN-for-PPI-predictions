clear all
clc
load yeast_data
num1=numel(P_protein_a);
num2=numel(N_protein_a);
result_1=[];
result_11=[];
result_2=[];
result_22=[];
N=60;
for i=1:num1
    result1=Laplace1(P_protein_a{i},N,'y');%ÕıĞò±àÂë
%     result1_inv=Laplace1(P_protein_a{i}(end:-1:1),N,'n');%ÄæĞò±àÂë
    result11=Laplace1(P_protein_b{i},N,'y');
%     result11_inv=Laplace1(P_protein_b{i}(end:-1:1),N,'n');
%     result_1=[result_1;[result1,result1_inv]];
    result_1=[result_1;result1];
    result1=[];
%     result_11=[result_11;[result11,result11_inv]];
    result_11=[result_11;result11];
    result11=[];
end
for i=1:num2
    result2=Laplace1(N_protein_a{i},N,'y');
%     result2_inv=Laplace1(N_protein_a{i}(end:-1:1),N,'n');
    result22=Laplace1(N_protein_b{i},N,'y');
%     result22_inv=Laplace1(N_protein_b{i}(end:-1:1),N,'n');
%     result_2=[result_2;[result2,result2_inv]];
    result_2=[result_2;result2];
    result2=[];
%     result_22=[result_22;[result22,result22_inv]];
    result_22=[result_22;result22];
    result22=[];
end
Pa=result_1;
Pb=result_11;
Na=result_2;
Nb=result_22;
P=[ones(length(P_protein_a),1),Pa,Pb];
N=[zeros(length(N_protein_a),1),Na,Nb];
data_SWA=[P;N];
save yeast_Laplace.mat data_SWA
