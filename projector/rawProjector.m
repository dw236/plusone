function G=rawProjector(filename,numDoc,vocSize,dim,alpha,beta)

fid=fopen(filename);
doc = zeros(numDoc,vocSize);

avgL=0;
for i=1:numDoc

    d = str2num(fgetl(fid));
    for j=1:length(d)
        doc(i,d(j)+1)=doc(i,d(j)+1)+1;

    end
    avgL=avgL+sum(doc(i,:));
    doc(i,:)=doc(i,:)/sum(doc(i,:));

end
avgL=avgL/numDoc;
fclose(fid);
% centroid = sum(doc)/size(doc,1);
% for i=1:numDoc
%     doc(i,:)=doc(i,:)-centroid;
% end
centroid = 0;

[ID, centroids, sums] = kmeans(doc, dim, 'replicates', 50, 'start', ...
                                 'cluster', 'distance', 'cosine', ...
                                 'EmptyAction', 'singleton');


centers = zeros(dim, vocSize);
sizes = zeros(1,dim);
for i=1:numDoc
    sizes(ID(i))=sizes(ID(i))+1;
    centers(ID(i),:)=centers(ID(i),:)+doc(i,:);
end
for i=1:dim
    centers(i,:)=centers(i,:)/sizes(i);
end
centers = centers - repmat(mean(centers), dim, 1); %dim(k, m)

bases = orth(centers'); %dim(m, k-1)
docProj = doc * bases; %dim(n, k-1)
centers = centers * bases; %dim(k, k-1)

% [U,S,V]=svds(doc,dim-1);
% docProj=U*S;
% 
% [ID,C,sumd]=kmeans(docProj,dim,'replicates',50,'start','cluster',...
%                    'distance','cosine', 'EmptyAction', 'singleton');
%
% centers = zeros(dim,dim-1);
% sizes = zeros(1,dim);
% for i=1:numDoc
%     sizes(ID(i))=sizes(ID(i))+1;
%     centers(ID(i),:)=centers(ID(i),:)+docProj(i,:);
% end
% for i=1:dim
%     centers(i,:)=centers(i,:)/sizes(i);
% end

hyperPlane = zeros(dim,dim-1);
bounds=zeros(1,dim);
for i=1:dim
    B=[centers(1:i-1,:);centers(i+1:dim,:)];
    X=zeros(dim-2,dim-1);
    for j=1:dim-2
        X(j,:)=B(j+1,:)-B(1,:);
    end
    hyperPlane(i,:)=null(X);
    bounds(i)=hyperPlane(i,:)*B(1,:)';
    if bounds(i)<0
        bounds(i)=-bounds(i);
        hyperPlane(i,:)=-hyperPlane(i,:);
    end
    if centers(i,:)*hyperPlane(i,:)' >0
        disp('the direction of the normal vector is wrong');
    end
end
X=docProj*hyperPlane';
for i=1:numDoc
    X(i,:)=X(i,:)./bounds;
end
scale=max(X,[],2);
k=floor(numDoc*(1-alpha));
for i=1:k+1
    [m,in]=max(scale);
    scale(in)=0;
end
centers=centers*max(m,1)/beta;
G=zeros(dim,vocSize);

for i=1:dim
    G(i,:)=max(centers(i,:)*bases'+centroid,0);
    G(i,:)=G(i,:)/sum(G(i,:));
end

Beta = log(G + 1e-323);
dlmwrite('data/final.beta', Beta, ' ');
end
