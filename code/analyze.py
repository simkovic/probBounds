import numpy as np
import pylab as plt
from scipy import stats
import os
import os.path as pth
from matusplotlib import errorbar,figure,saveStanFit, loadStanFit,plotCIwald,plotCIttest2,pearsonrCI,subplotAnnotate
np.set_printoptions(suppress=True,precision=3)
DPI=400
PATH=pth.abspath(pth.join(os.getcwd(), pth.pardir))
FPATH=pth.join(PATH,'publication','figs','')
SEED=11
tp='anon'
OPATH=pth.join(PATH,'output',tp)+pth.sep


def checkData():
    #load metadata 
    info=np.int32(np.loadtxt(OPATH+'vpinfo.nfo',delimiter=','))

    print('Check if all data files present')
    for i in range(info.shape[0]):
        fn='vp%03d.'%info[i,0]
        for suf in ['res','log']:
            if not pth.isfile(OPATH+fn+suf):print(fn+suf+' is missing')
    print('Checking for surplus files, with no meta-data') 
    fnsall=os.listdir(OPATH)
    fns=list(filter(lambda x: x.endswith('.res'),fnsall))
    fns=fns+list(filter(lambda x: x.endswith('.log'),fnsall))
    fns=np.sort(fns)
    for fn in fns:
        vp=int(fn.rsplit('.')[0].rsplit('vp')[1])
        temp=(info[:,0]==vp).nonzero()[0]
        if not temp.size==1: print(fn+' surplus file')
            
    print('Done')
    
def loadData(verbose=False): 
    ccccc=[]
    print('Loading data')
    info=np.int32(np.loadtxt(OPATH+'vpinfo.nfo',delimiter=','))         
    fnsall=os.listdir(OPATH)
    fns=list(filter(lambda x: x[-4:]=='.res',fnsall))
    K=len(fns)
    D=[-np.ones((K,7,2)),
       np.nan*np.zeros((K,5,11)), 
       np.nan*np.zeros((K,5,5)),
       np.nan*np.zeros((K,6)),
       np.nan*np.zeros((K,11)),
       np.nan*np.zeros((K,3,11))]
    #,[None,None,0],np.nan*np.zeros((6,2))])
    nfos=[]
    dknow=[]
    for k in range(K):
        fn=fns[k]
        dknow.append([0,0])
        vp=int(fn.rsplit('.')[0].rsplit('vp')[1])
        nfo=info[(info[:,0]==vp).nonzero()[0][0],:]
        nfos.append(nfo)
        if verbose: print(fn,nfo)

        try: 
            f=open(OPATH+fn,'r', encoding='iso 8859-15')
        except:pass
        lns=f.readlines()
        f.close()
        version=int(lns[0][0])
        lns[0]=lns[0][2:]
        assert(version>=0)
        ccccc.append([vp,version])
        
        lns=list(map(lambda x: x.rstrip('\n'),lns))
        temp=list(map(int,lns[0].rsplit(',')[:-1]))
        assert(len(temp)==[26,28,19,21][version])
        tl=[3,10][int(version<2)]
        for i in range(tl):# equivalenzklassen 3 o 10
            ixs=[[[0,1],[0,2],[1,2]],[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]][int(version<2)][i]
            if temp[i]==1:  D[2][k,ixs[0],ixs[1]]=D[2][k,ixs[1],ixs[0]]=1
            elif temp[i]==2: D[2][k,ixs[0],ixs[1]]=D[2][k,ixs[1],ixs[0]]=0
        #gleich w'lich 1
        dknow[-1][1]+=1
        if temp[tl]==1: D[0][k,6,0]=1
        elif temp[tl]==2: D[0][k,6,0]=0
        elif temp[tl]==3: dknow[-1][0]+=1
        else:dknow[-1][1]-=1
        
        for i in range(6):#w'er Beispiele 6
            dknow[-1][1]+=1
            if temp[i+tl+1]==3:D[3][k,i]=0
            elif temp[i+tl+1]==1: D[3][k,i]=1
            elif temp[i+tl+1]==2: D[3][k,i]=-1
            elif temp[i+tl+1]==4: dknow[-1][0]+=1
            else:dknow[-1][1]-=1
            
        for i in range([9,11][int(version==1)+int(version==3)]):
            dknow[-1][1]+=1
            sh=[0,2][int(i>4 and version==2)]
            if temp[i+tl+7]==1: D[4][k,i+sh]=1
            elif temp[i+tl+7]==2: D[4][k,i+sh]=0
            elif temp[i+tl+7]==3: dknow[-1][0]+=1
            else:dknow[-1][1]-=1
              
        if verbose: print('\tline 1 ok')
        for i in range(1,6):
            if lns[i]=='-':continue
            if len(lns[i].rsplit(','))==1: 
                D[1][k,i-1,:]=0
            else:
                temp=np.array(list(map(int,lns[i].rsplit(','))))
                assert(np.all(temp>0))
                assert(np.all(temp<11))
                D[1][k,i-1,:]=0
                D[1][k,i-1,temp-1]=1
                if 10 in set(temp) and nfo[-1]==0: D[1][k,i-1,10]=1
            if nfo[-1]==0:D[1][k,i-1,9]=np.nan
            else: D[1][k,i-1,10]=np.nan 
            if verbose: print('\tline %d ok'%(i+1))
        def parseOP(ln):
            els=ln.rsplit(',')
            assert(len(els)==10)
            temp=np.zeros(11)*np.nan
            for ei in range(len(els)):
                nms=els[ei].rsplit(' ')
                for nm in nms: 
                    if len(nm):
                        if int(nm)==10 and nfo[-1]==0:  temp[10]=ei
                        else: temp[int(nm)-1]=ei
            return temp
        for i in range(2):
            temp=lns[6+i].replace(' ','')
            if len(temp): D[5][k,i,:]=parseOP(lns[6+i])
            elif i>0: D[5][k,i,:]=D[5][k,i-1,:]
            if verbose: print('\tline %d ok'%(i+7))
        D[5][k,:,:]/=10
        D[5][k,:,:]=1-D[5][k,:,:]
        if len(''.join(lns[8].rsplit(',')))==0:temp=[-1]*10
        else:
            temp=list(map(lambda x: min(float(x)/142,1),lns[8].rsplit(',')))
            assert(len(temp)==10)
            assert(np.all(np.array(temp)>=0))
            assert(np.all(np.array(temp)<=1))
        D[5][k,2,:9]=temp[:-1]
        if nfo[-1]==0: D[5][k,2,10]=temp[-1]
        else: D[5][k,2,9]=temp[-1]
        if verbose: print('\tline %d ok'%(9))
        for i in range(9,15):
            if lns[i]=='-':continue
            temp=lns[i].replace(' ','')
            a,b=temp[1:-1].rsplit('\',\'')
            D[0][k,i-9,0]=len(a)>0
            D[0][k,i-9,1]=len(b)>0
            if verbose: print('\tline %d ok'%(i+1))
    print('\tformat of all .res files is ok')
    nfos=np.array(nfos)
    ag=info[:,2]/360
    D.append(ag)
    print('\tage: mean= %.1f y, median= %.1f, min= %.1f, max= %.1f, girls=%d):'%(ag.mean(),np.median(ag),np.min(ag),np.max(ag),info[:,1].sum()))
    np.savetxt('ccccc',ccccc,fmt='%d')
    dknow=np.array(dknow)
    #print(dknow)
    print(dknow[:,0].sum(),' said dont know out of ',dknow[:,1].sum(),' questions')
    return D
      
checkData()
D=loadData()

def list2d2latextable(lst,decim=2,header=None,colheader=None):
    ''' array - 2D numpy.ndarray with shape (rows,columns)
        decim - decimal precision of float, use 0 for ints
            should be int scalar or a list of list,len(decim)=nr cols
    '''
    assert(np.all(np.array(list(map(len,lst)))==len(lst[0])))
    ecol=' \\\\\n';shp=(len(lst),len(lst[0]))
    out='\\begin{table}\n\\centering\n\\begin{tabular}{|'+ \
        ['l|',''][int(colheader is None)]+shp[1]*'r|'+'}\n\\hline\n'
    if not header is None:
        s=len(header)*'{} & '
        out+=s.format(*header)[:-2]+ecol+'\\hline\n'   
    for i in range(shp[0]):
        if not colheader is None:
            out+='%s & '%colheader[i]
        for j in range(shp[1]):
            if type(decim) is list: dc=decim[j]
            else: dc=decim
            if type(lst[i][j])==str:
                out+='%s'%lst[i][j]
            elif dc==0: out+='%d'%int(lst[i][j])
            elif np.isnan(lst[i][j]): out+=' '
            else:
                flt='{: .%df}'%dc
                out+=flt.format(np.round(lst[i][j],dc))
            if j<shp[1]-1: out+=' & '
        out+=ecol
    out+='\\hline\n\\end{tabular}\n\\end{table}'
    print(out)   

print(20*'#','\nprint latex code for Table 2 \n\n')################# 
wlbl=['möglich','unmöglich','sicher','wahrscheinlich','unwahrscheinlich','wahrscheinlicher als'] 
wlbl=['possible','impossible','certain','likely','unlikely','likelier than'] 

ch=['Wort schon gehört','Beispiel genannt','Erklärung genannt','Beispiel und Erklärung','Beispiel oder Erklärung']
ch=['Had heard','Gave example','Provided explaination','Example \\& explanation','Example or explanation']

T=[]
T.append(list(100*np.mean(D[0][:,:6,0]>-1,0)))
T.append(list(100*np.mean(D[0][:,:6,1]==1,0)))
T.append(list(100*np.mean(D[0][:,:6,0]==1,0)))
T.append(list(100*np.mean(np.logical_and(D[0][:,:6,0]==1,D[0][:,:6,1]==1),0)))
T.append(list(100*np.mean(np.logical_or(D[0][:,:6,0]==1,D[0][:,:6,1]==1),0)))
list2d2latextable(T,decim=0,header=[' ']+wlbl,colheader=ch)

print('gleichwahrscheinlich',np.round((D[0][:,6,0]==1).mean(),3))
print(20*'##','\nprint latex code for Table 3 \n\n')################# 

   
bsp=['feuerwerk','schnee','rote ampel','1fc köln','blaue ampel','regen','sonne',
     'fischstäbchen','hagel','schule WT','schule WE']
bsp=['Fireworks','Snow','Red traffic light','Soccer team','Blue traffic light','Rain','Rising sun',
     'Fish sticks','Hail','School WD','School WE']
bspind=[4,0,3,7,1,8,5,2,6,9,10]
bspsorted=list(np.array(bsp)[bspind])
T=np.nanmean(D[1],0)*100
list2d2latextable(T[:,bspind].T,decim=0,header=[' ']+wlbl[:-1],colheader=bspsorted)  
print(20*'##','\nprint table with class overlap (unpublished  \n\n')################# 

 
T=-np.ones((5,5))
T[:-1,1:]=np.nanmean(D[2][:,:-1,1:],0)*100

T[np.isnan(T)]=-1
T = np.maximum( T, T.transpose())
#T[T==-1]=0
list2d2latextable(T,decim=0,header=[' ']+wlbl[:-1],colheader=wlbl[:-1])

#print('sample size',(~np.isnan(D[2][:,:-1,1:])).sum(0))

# plot figure S1
a=np.nanmean(D[3]==1,0)
b=np.nanmean(D[3]==0,0)
c=np.nanmean(D[3]==-1,0)
axL = plt.subplot(1,1,1)
plt.barh(np.arange(D[3].shape[1]),width=a,color='r')
plt.barh(np.arange(D[3].shape[1]),width=b,left=a,color='y')
plt.barh(np.arange(D[3].shape[1]),width=c,left=a+b,color='k')
plt.xlim([0,1])
axL.set_xticks(np.linspace(0,1,11))
#axL.set_yticklabels(['','regen','ampel rot','regen','sonne','schnee','sonne'],color='r')
axL.set_yticklabels(['','rain','red tr. light','rain','rising sun','snow','rising sun'],color='r')
plt.grid(True,axis='x')
plt.ylim([-0.5,5.5])
axR = plt.gca().twinx()
axR.yaxis.tick_right()
axR.set_yticks(np.arange(D[3].shape[1]))
plt.ylim([-0.5,5.5])
#axR.set_yticklabels(['schnee','ampel blau','hagel','ampel rot','hagel','ampel blau'],color='g');
axR.set_yticklabels(['snow','blue tr. light','hail','red tr. light','hail','blue tr. light'],color='k');
plt.savefig(FPATH+'S1_comparison.png')
#check transitivity: index 5 non-transitive
# check consistency with line order data

ind=[[5,1],[2,4],[5,8],[6,2],[1,8],[6,4]]
K=np.zeros((5,5,3))
for i in range(D[1].shape[1]):
    for j in range(D[1].shape[1]):
        for k in range(D[3].shape[1]):
            for n in range(D[1].shape[0]):
                if D[1][n,i,ind[k][0]]*D[1][n,j,ind[k][1]]:
                    if not np.isnan(D[3][n,k]): K[i,j,int(D[3][n,k])+1]+=1
#evaluate and plot results from line allocation task (unpublished results)
L=np.zeros(K.shape)
flt=np.ones(K.shape[0])-np.identity(K.shape[0])
L[:,:,0]=K[:,:,0]+K[:,:,2].T
L[:,:,1]=K[:,:,1]+K[:,:,1].T
L[:,:,2]=K[:,:,2]+K[:,:,0].T
B=np.copy(L.sum(2))
L=L/B[:,:,np.newaxis]    
    
axL = plt.gca()
a=L[:,:,0].flatten()
b=L[:,:,1].flatten()
c=L[:,:,2].flatten()
d=np.arange(K.shape[0]*K.shape[1])
plt.barh(d,width=a,color='r')
plt.barh(d,width=b,left=a,color='k')
plt.barh(d,width=c,left=a+b,color='g')
plt.xlim([0,1])
#axL.set_xticks(np.linspace(0,1,))
wlbl=['possible','impossible','certain','likely','unlikely']
wlbl=np.array(len(wlbl)*[wlbl])
axL.set_yticks(d)
axL.set_yticklabels(wlbl.flatten().tolist(),color='r')
plt.grid(True,axis='x')
axL = plt.gca()
a=L[:,:,0].flatten()
b=L[:,:,1].flatten()
c=L[:,:,2].flatten()
d=np.arange(K.shape[0]*K.shape[1])
plt.barh(d,width=a,color='r')
plt.barh(d,width=b,left=a,color='k')
plt.barh(d,width=c,left=a+b,color='g')
plt.xlim([0,1])
#axL.set_xticks(np.linspace(0,1,))
wlbl=['possible','impossible','certain','likely','unlikely']
wlbl=np.array(len(wlbl)*[wlbl])
axL.set_yticks(d)
axL.set_yticklabels(wlbl.flatten().tolist(),color='r')
plt.grid(True,axis='x')
plt.ylim([-0.5,24.5]);
axR = plt.gca().twinx()
axR.yaxis.tick_right()
axR.set_yticks(d)
axR.set_yticklabels(wlbl.T.flatten().tolist(),color='g');
plt.ylim([-0.5,24.5]);
plt.ylim([-0.5,24.5]);
axR = plt.gca().twinx()
axR.yaxis.tick_right()
axR.set_yticks(d)
axR.set_yticklabels(wlbl.T.flatten().tolist(),color='g');
plt.ylim([-0.5,24.5]);
print(20*'##','\nclass transitivity and overlap with prespecified events (results mentioned in publications text) \n\n')################# 

#intransitivity with prespecified event
C=np.array([[[1,1,1],
     [1,0,0],
     [1,0,0]],
    [[1,0,0],
     [0,1,0],
     [0,0,1]],
    [[0,0,1],
     [0,0,1],
     [1,1,1]]])
d=np.int32(D[3])+1
out=[[],[]]
for i in range(d.shape[0]):
    for j in range(2):
        if np.any(d[i,[0+j,2+j,4+j]]<0): out[j].append(np.nan)
        else: out[j].append(C[d[i,0+j],d[i,2+j],d[i,4+j]])
out=np.array(out)
print(f'intransitivity with prespecified event: {(out==0).sum()} out of {out.size}')
K=np.zeros((5,5))
for i in range(5):
    for j in range(5):
        a=D[1][:,i,:].flatten();b=D[1][:,j,:].flatten()
        c=(np.nansum(D[1][:,j,:],1)>0).sum()
        sel=np.logical_and(~np.isnan(a),~np.isnan(b))
        K[i,j]=np.logical_and(a[sel],b[sel]).sum()/b[sel].sum()
        if i==0 and j==1:print('p(possible|impossible)',b[sel].sum(),c, np.round(plotCIwald(K[i,j],df=b[sel].sum()),3))
        if i==0 and j==2:print('p(possible|certain)',b[sel].sum(),c, np.round(plotCIwald(K[i,j],df=b[sel].sum()),3)) 
        if i==1 and j==4:print('p(impossible|unlikely)',b[sel].sum(),c, np.round(plotCIwald(K[i,j],df=b[sel].sum()),3)) 
        if i==4 and j==1:print('p(unlikely|impossible)',b[sel].sum(),c, np.round(plotCIwald(K[i,j],df=b[sel].sum()),3)) 
        #K[i,j]=np.corrcoef(a[sel],b[sel])[0,1]

#print(f'conditional probability p(moglich|unmoglich) \n',K)
print(20*'##','\nevaluate answers to conceptual-knowledge questions, plot figure 1A \n\n')################# 
 
lbl=['transitivität erwartet','transitivität zwingend','alle möglichen gleich',
     'alle unmöglichen gleich','alle sicheren gleich','total wahrscheinlich',
     'total unwahrscheinlich','wahrscheinlicher als sicher',
     'unwahrscheinlicher als unmöglich','ordnung transitiv','linie transitiv']
labels=['knows equally likely','possible & impossible\noverlap',
    'possible & certain\noverlap','impossible & certain\noverlap',
    'possible & likely\noverlap','impossible & unlikely\noverlap',
    'prespecified comparison\nis transitive','transitivity is necessary\n','all possible E are\nequally likely',
    'all impossible E are\nequally likely','all certain E are\nequally likely','P increases\ninfinitely',
    'P decreases\ninfinitely','E more likely than\ncertain E exists','E less likely than\nimpossible E exists',
    'ranking is transitive\n','line is transitive\n']
#print(D[4]);
ordr=np.array([5,14,15,6,8,9,7,0,1,2,3,4,10,11,12,13])[::-1]
print((D[4]==2).sum(),' said dont know from all answered questions',(~np.isnan(D[4])).sum())
D[4][D[4]==2]=np.nan

sel=~np.logical_and(np.isnan(D[4][:,5]),np.isnan(D[4][:,6]))
print('no infinite increse/decrease',np.round(plotCIwald(np.logical_and(D[4][sel,5],D[4][sel,6]))[:3],2))
print('no E beyond bounds',np.round(plotCIwald(np.logical_and(1-D[4][sel,7],1-D[4][sel,8]))[:3],2))

temp=np.reshape(np.rollaxis(D[0][:,:3,:],2,1),[D[0].shape[0],-1])
temp[temp==-1]=0
#age=np.int32(info[:,2]>8*365)
gw=np.copy(D[0][:,6,0])
gw[gw==-1]=0
sel=np.logical_and(D[2][:,0,3]==1,np.isnan(D[2]).sum(2).sum(1)==5)
ddd=np.int32(D[2][sel,2,:][:,[0,1,4]]==D[2][sel,3,:][:,[0,1,4]])
dd=ddd.sum(1)
print('guessing check',sel.sum(),(dd==3).sum(),(ddd==0).mean(),ddd.size)
temp2=np.vstack([gw,D[2][:,0,1],D[2][:,0,2],D[2][:,1,2],D[2][:,0,3],D[2][:,1,4]]).T
Y=np.concatenate([temp2,D[4]],axis=1) 
tmp=np.array([0,0,0,0,0,0, 0,0, 0,0,0, 1,1, 0,0, 0,0],ndmin=2)#invert some items, see labels
Y= (1-tmp)*Y +tmp*(1-Y)
Y=np.abs(Y)
figure(size=3,dpi=400)
plt.subplot(1,2,1)

out=plotCIwald(Y[:,1:][:,ordr],verticalErrorbar=False,clr='k',alpha2=None)
plt.grid()
ax=plt.gca()
ax.invert_yaxis()
ax.spines['left'].set_visible(False)
plt.ylim(-.5,len(labels[1:])-.5)
ax.set_yticks([])
ax.tick_params(labeltop=True)
lbls=[]
N=np.sum(~np.isnan(Y),0)
for k in range(1,N.size):
    lbls.append(labels[k]+' (n='+str(N[k])+')')

#ax.set_yticks(np.arange(len(lbls)))
#ax.set_yticklabels(lbls)
plt.xlim([0,1])
ax.set_xticks(np.linspace(0,1,11))
plt.xlabel('Proportion of children\nwho agreed with statement')
subplotAnnotate()
#plt.savefig(FPATH+'ck.png',dpi=DPI,bbox_inches='tight')
#list2d2latextable(np.array([T2,T3,T*100],ndmin=2).T,decim=0,colheader=labels)  

print(20*'##','\nevaluate effect of age, plot figure 1B \n\n')
def ageRegression(age,y,labels,run=True):
    ''' age - age predictor in years
        y - binary outcome (missing vals as nan)
    '''
    
    logreg="""
    data {
      int<lower=0> N;
      int<lower=0> D;
      matrix[N,D] x;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      vector[D] beta;
    }
    model {
      y ~ bernoulli_logit(x*beta);
    }

    """
    if run:
        import pystan
        sm=pystan.StanModel(model_code=logreg)  
    x=np.array([np.ones(age.shape[0]),age-8.5]).T
    out=[]
    for d in range(min(y.shape[1],14)):   
        sel=~np.isnan(y[:,d])
        if run:
            fit=sm.sampling(data={'N':sel.sum(),'D':x.shape[1],
                'y':np.int32(y[sel,d]),'x':x[sel,:]},chains=6,
                n_jobs=6,seed=SEED,thin=5,iter=10000,warmup=5000)  
            assert(np.all(fit.summary()['summary'][:2,-1]<1.05))
            saveStanFit(fit,fname=f'standata/agereg{d}')
        w=loadStanFit(f'standata/agereg{d}')
        out.append(w['beta'])
        
    out=np.array(out)
    #figure(size=2,dpi=400)
    plt.subplot(1,2,2)
    ax=plt.gca()
    #ax.invert_yaxis()
    plt.grid(True,axis='x');plt.grid(False,axis='y')
    x=np.array([8,7,6,15,12,9,11,10,3,2,1,0])
    ax.set_yticklabels([])
    ax.set_yticks(np.arange(len(labels)))
    #print(out.shape,x.shape);stop
    errorbar(out[1:-1,:,1].T,x=x,clr='k',          
        pi2=None,verticalErrorbar=False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(np.array(labels)[ordr]);
    for tick in ax.yaxis.get_major_ticks():tick.set_pad(40)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_horizontalalignment("center")
    plt.ylim([-0.5,len(labels)-.5])
    ax.set_xticks(np.linspace(-2,1.5,8))
    ax.tick_params(labeltop=True)
    for d in range(out.shape[0]):
        print(labels[d]+'\t',np.round(np.median(1/(1+np.exp(-out[d,:,0]+1.5*out[d,:,1]))),3),
        np.round(np.median(1/(1+np.exp(-out[d,:,0]-out[d,:,1]*1.5))),3),
        errorbar(out[d,:,1],plot=False,pi2=None,verticalErrorbar=False))
    plt.xlabel('Change in log odds when\nchild becomes one year older')
    
print('correlation sumCorrect:age = (r,95%CI l,95%CI u)',pearsonrCI(np.nanmean(Y[:,1:13],1),D[6]))
print('statement (n=children) proportion yes at 7y, proportion at 10y, [[log-odds per year, 95% interval low, 95% interval up]]')
ageRegression(D[6],Y,lbls,run=False)
subplotAnnotate()
plt.gcf().tight_layout()
plt.savefig(FPATH+'ck.tif',dpi=DPI,bbox_inches='tight')

print(20*'##','\nMCA\n\n')################# 
seld=list(range(1,9))+[11,12]
lbls=np.array(labels)[seld]
X=Y[:,seld]
sel=~np.any(np.isnan(X),axis=1)
X=X[sel,:]
X=np.hstack([X,1-X])
assert(np.all(np.logical_or(X==0,X==1)))
X=np.int32(X)
print(f'MCA sample size:{X.shape}')
import pandas as pd
import mca
mc=mca.MCA(pd.DataFrame(data=X))
fs=mc.fs_c(N=X.shape[1]//2)[:X.shape[1]//2,:]
ev=mc.expl_var(greenacre=True)
print(f'MCA % explained variance: {ev*100}')
N=3
print(ev[N:].sum(),ev.sum())
figure(size=2,dpi=400)
plt.plot(fs[:,:N],-np.arange(len(lbls)),'-o')
ax=plt.gca()
plt.plot([0,0],[1,-12],'k')
ax.set_yticks(-np.arange(len(lbls)));
plt.ylim([-len(lbls)+.5,0.5]);plt.xlim([-.8,.4])
plt.grid(True)
ax.set_yticklabels(lbls);
plt.legend(range(1,N+1))
plt.savefig(FPATH+'S3_mca.png',dpi=DPI,bbox_inches='tight')
# age:1stPC
for n in range(N):
    print(f'{n+1}PC:age; r, clL, clU, p-val',np.round(pearsonrCI(mc.fs_r(N=20)[:,n],-D[6][sel]),2))
    
    
sll=Y[sel,10]
sll=D[0][sel,6,0]
pcs=mc.fs_r(N=20)
for i in range(3):
    #plotCIttest2(pcs[:,i][sll==1],pcs[:,i][D[0][sel,6,0]==0],x=i)
    d=pcs[:,i][sll==1].mean()-pcs[:,i][sll==0].mean()
    ss=(pcs[:,i][sll==1].var()/2+pcs[:,i][sll==0].var()/2)**0.5
    
    print(f'{i}th PC: cohens d difference between children who knew eq. likely and who didnt',d/ss)
    
for k in range(2):
    al=D[4][:,3+k]==1
    w=['unmoglich','sicher'][k]
    print(f'{int(np.nansum(D[4][al,8-k]==0))} say w_er als {w} exists among {int(np.nansum(al))}'+
         f' who say {w} is equivalence class')
print(20*'##','\n\n')#################         
