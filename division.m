clear;
clc;
[numF,txtF,rawF]=xlsread('Files');
i=1;
for b=1:24
    count=0;
    for a=1:length(rawF)
        file=rawF(a,b);
        if (~isnan(file{1}))
            NameFile=append(file{1},'.edf');
            [numS,txtS,rawS] = xlsread('Seizures');
            idx = find(strcmp(rawS, NameFile));
            if (~isempty(idx))
                data= edfread(NameFile);
                info= edfinfo(NameFile);
                fs = info.NumSamples/seconds(info.DataRecordDuration);
                signum = 22; %sensor number
                t = (0:info.NumSamples(signum)-1)/fs(signum);
                for recnum =1:(info.NumDataRecords) %line number (second number)
                    %State(i)="normal";
                    flag=0;
                    for j=1:length(idx)
                        if ((numS(idx(j)-1,1)<=recnum)&&(numS(idx(j)-1,2)>=recnum))
                            %State(i)="ictal";
                            flag=1;
                            break;
                        elseif ((numS(idx(j)-1,1)>=recnum)&&(numS(idx(j)-1,1)-20<=recnum)) 
                            %State(i)="preictal";
                            flag=1;
                            break;
                        end
                    end
                    Flag(i)=flag;
                    MeanVector(i)=mean(data.(signum){recnum});
                    VarVector(i)=(std(data.(signum){recnum}))^2;
                    SDVector(i)=std(data.(signum){recnum});
                    AmplitudeVector(i)=range(data.(signum){recnum});
                    i=i+1;
                end
            end
        end
    end
end
NewData=[MeanVector',VarVector',SDVector',AmplitudeVector',Flag'];
matObj = matfile('MDP_finalize','Writable',true);
save('MDP_finalize','NewData');