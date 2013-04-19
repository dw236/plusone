function G=cooccurrence(filename,numDoc,vocSize,dim, alpha)

fid=fopen(filename);
doc = zeros(numDoc,vocSize); %dim(n, m)

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
centroid = sum(doc)/size(doc,1);
for i=1:numDoc
    doc(i,:)=doc(i,:)-centroid;
end

C = doc' * doc; %dim(m, n) * dim(n, m) = dim(m, m)
baseline = sum(doc); %dim(1, m)
baseline = baseline / sum(baseline);
B = baseline' * baseline; %dim(m, m)

D = C - alpha * B; %dim(m, m)

[U,S,V]=svds(D,dim-1);
docProj=doc * V;

G = docProj;
