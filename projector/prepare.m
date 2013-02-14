function prepare(filename,numDoc,vocSize,dim)

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
centroid = sum(doc)/size(doc,1);
for i=1:numDoc
    doc(i,:)=doc(i,:)-centroid;
end

[U,S,V]=svds(doc,dim-1);
docProj=U*S;

dlmwrite('data/projected', docProj, ' ');
dlmwrite('data/centroid', centroid, ' ');
dlmwrite('data/V', V, ' ');
end