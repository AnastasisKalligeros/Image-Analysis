close all
clc
clear



%loading the Math Works Merch Dataset
unzip('MerchData.zip');
imds = imageDatastore('MerchData','IncludeSubfolders',true, 'LabelSource','foldernames'); 
images={};
pointer=1;
for i=1:length(imds.Files)
    images{pointer}=imread(char(imds.Files(i)));
    pointer=pointer+1;
end

%loading a pretrained resnet18 model
net=resnet18;
input_size=net.Layers(1).InputSize;
extract_layer='pool5';

%feature extraction
features=zeros(75,512);
for i=1:length(images)
    features(i,:)=activations(net,imresize(images{i},[input_size(1),input_size(2)]),extract_layer,'OutputAs','rows');
end

%distance ranking for each image and rank normalization
T=zeros(75,75);
for i=1:size(features,1)
    for j=1:size(features,1)
        T(i,j)=sqrt(sum(features(i,:)-features(j,:)).^2);
    end
end

for i=1:size(T,1)
    for j=1:size(T,2)
        T(i,j)=(T(i,j)+T(j,i))/2;
    end
end

%hypergraph construction for 5 most similar images
sorted_indexes=zeros(75,75);
for i=1:75
    [~,sorted_indexes(i,:)]=sort(T(i,:));
end

sorted_indexes=sorted_indexes(:,1:5);
G=zeros(75,75);

%log based distance calculation
for i=1:75
    for j=1:5
        position=1-(log(j)/log(6));
        G(i,sorted_indexes(i,j))=position;
    end
end

%similarities calculation
Sh=G*G';
Sv=G'*G;
S=Sh.*Sv;


%cartesian product calculation

%hyperedge cummulative weight
w=zeros(75,1);
for i=1:75
    w(i)=sum(G(i,:));
end

%cartesian product calculation
C=zeros(75,75);
for i=1:75
    for j=1:75
        C(i,j)=w(i)*G(i,j);
    end
end

%final weights calculation

W=C.*S;

similar_images=zeros(75,5);

sorted_indexes=zeros(75,75);
for i=1:75
    [~,sorted_indexes(i,:)]=sort(W(i,:),'descend');
end

similar_images=sorted_indexes(:,1:5);

%plotting 4 images and their 4 most similar images
random_examples=randi(75,[1,4]);


for i=1:4
    img=random_examples(i);
    figure
    subplot(1,5,1);
    imshow(images{similar_images(img,1)});
    title('Target Image');
    for i=2:5
        subplot(1,5,i)
        imshow(images{similar_images(img,i)});
    end
end


figure("Name","RANK NORMALIZATION")
heatmap(T)
figure("Name","HYPERGRAPH BASED SIMILARITY")
heatmap(W)
