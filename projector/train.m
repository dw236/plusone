function train(numDoc, vocSize, dim, alpha, beta)

centroid = load('data/centroid');
docProj = load('data/projected');
V = load('data/V');
ID = load('data/labels');

% centers = zeros(dim,dim-1);
% sizes = zeros(1,dim);
% for i=1:numDoc
%     sizes(ID(i))=sizes(ID(i))+1;
%     centers(ID(i),:)=centers(ID(i),:)+docProj(i,:);
% end
% for i=1:dim
%     centers(i,:)=centers(i,:)/sizes(i);
% end

centers = load('data/centers');

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
    G(i,:)=max(centers(i,:)*V'+centroid,0);
    G(i,:)=G(i,:)/sum(G(i,:));
end

Beta = log(G + 1e-323);
dlmwrite('data/final.beta', Beta, ' ');

end