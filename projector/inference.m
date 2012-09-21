function P=inference(model, testingSet, numDocs, vocSize, numTopics)
%model: final.beta (should be 'data/final.beta')
%testingSet: test_documents (should be found in 'data/test_documents')

% Read the model into V, each row is a topic, the entries are probabilities
% (not log probabilities)
V = load(model); %dim(numTopics, vocSize)
V = exp(V);

% Read the testingDoc into D
fid = fopen(testingSet);
D=zeros(numDocs,vocSize);
for i=1:numDocs
    d = str2num(fgetl(fid));
    for j=1:length(d)
        index = d(j) + 1;
        D(i, index) = D(i, index) + 1;
    end
end

[L,U]=lu(V');
T=U\(L\D');
P=T'*V;

% Write P to a file so java can read it. P is (numDoc)*(vocSize), P_ij is
% the probability of word j in testing document i. In java just read the
% matrix and return it as result.
dlmwrite('data/predictions', P, ' ');
