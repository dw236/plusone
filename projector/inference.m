function P=inference(model, testingSet,numDoc, vocSize, dim)

fid=fopen(model);
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