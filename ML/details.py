def details(cm):
    print(cm)
    if (len(cm)==3):
        precision0=(cm[0,0])/(cm[0,0]+cm[1,0]+cm[2,0])
        precision1=(cm[1,1])/(cm[0,1]+cm[1,1]+cm[2,1])
        precision2=(cm[2,2])/(cm[0,2]+cm[1,2]+cm[2,2])
        precision=(precision0+precision1+precision2)/3
        sensitivity0=(cm[0,0])/(cm[0,0]+cm[0,1]+cm[0,2]+cm[1,2]+cm[2,1])
        sensitivity1=(cm[1,1])/(cm[1,1]+cm[0,2]+cm[1,0]+cm[1,2]+cm[2,0])
        sensitivity2=(cm[2,2])/(cm[2,2]+cm[0,1]+cm[1,0]+cm[2,0]+cm[2,1])
        sensitivity=(sensitivity0+sensitivity1+sensitivity2)/3
        spec0=(cm[1,1]+cm[2,2])/(cm[1,0]+cm[2,0]+cm[1,1]+cm[2,2])
        spec1=(cm[0,0]+cm[2,2])/(cm[0,1]+cm[2,1]+cm[0,0]+cm[2,2])
        spec2=(cm[0,0]+cm[1,1])/(cm[0,2]+cm[1,2]+cm[0,0]+cm[1,1])
        spec=(spec0+spec1+spec2)/3
        FDR0=(cm[1,0]+cm[2,0])/(cm[1,0]+cm[2,0]+cm[0,0])
        FDR1=(cm[0,1]+cm[2,1])/(cm[0,1]+cm[1,1]+cm[2,1])
        FDR2=(cm[0,2]+cm[1,2])/(cm[0,2]+cm[1,2]+cm[2,2])
        FDR=(FDR0+FDR1+FDR2)/3
                        
        print("\n")
        print("precision:",precision*100)
        print("class 0:",precision0*100)
        print("class 1:", precision1*100)
        print("class 2 :" , precision2*100)
        print("\n")
        print("sensitivity:",sensitivity*100)
        print("class 0:",sensitivity0*100)
        print("class 1:",sensitivity1*100)
        print("class 2:", sensitivity2*100)
        print("\n")
        print("specificity", spec*100)
        print("class 0:",spec0*100)
        print("class 1:",spec1*100)
        print("class 2:",spec2*100)
        print("\n")
        print("FDR:",FDR*100)
        print(FDR0,FDR1,FDR2)

    elif(len(cm)==2):
        prec0=cm[0,0]/(cm[0,0]+cm[1,0])
        prec1=cm[1,1]/(cm[1,1]+cm[0,1])
        prec=(prec0+prec1)/2
        print("precision:",prec*100)
        sens=cm[0,0]/(cm[0,0]+cm[0,1])
        print("sensitivity:",sens*100)
        spec=cm[1,1]/(cm[1,0]+cm[1,1])
        print("specificity:",spec*100)
        FDR0=cm[1,0]/(cm[1,0]+cm[0,0])
        FDR1=cm[0,1]/(cm[0,1]+cm[1,1])
        FDR=(FDR0+FDR1)/2
        print("FDR:",FDR*100)
    
