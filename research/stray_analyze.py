import sys, numpy as np, pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
root, tag, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
o=pd.read_csv(f'{root}/odometry.csv',skipinitialspace=True); s=256/1920
p=o[['x','y','z']].values
print(tag,'frames',len(o),'dur s',round(o.timestamp.iloc[-1]-o.timestamp.iloc[0],1),
      'traj extent',np.round(p.max(0)-p.min(0),2),'start-end gap',round(np.linalg.norm(p[-1]-p[0]),3),
      'fx',o.fx.min(),o.fx.max(),'n_fx',o.fx.nunique())
pts=[]
for i in range(0,len(o),step):
    r=o.iloc[i]; f=int(r.frame)
    d=np.array(Image.open(f'{root}/depth/{f:06d}.png')).astype(np.float32)/1000
    c=np.array(Image.open(f'{root}/confidence/{f:06d}.png'))
    v,u=np.nonzero((c==2)&(d>0.2)&(d<5)); z=d[v,u]
    P=np.stack([(u+0.5-r.cx*s)/(r.fx*s)*z,(v+0.5-r.cy*s)/(r.fy*s)*z,z],1)
    pts.append(P@R.from_quat([r.qx,r.qy,r.qz,r.qw]).as_matrix().T+[r.x,r.y,r.z])
P=np.concatenate(pts)
h,e=np.histogram(P[:,1],bins=np.arange(P[:,1].min(),P[:,1].max(),0.01))
floor=e[np.argmax(h[:len(h)//2])]+0.005
H=P[:,1]-floor
hh,ee=np.histogram(H,bins=np.arange(1.8,4.0,0.01))
up=[(round(ee[i]+0.005,2),round(100*hh[i]/max(1,(H>1.8).sum()),1)) for i in np.argsort(hh)[::-1][:6]]
print(tag,'floor y',round(floor,3),'| upward pts share >1.8m: %.1f%%'%(100*(H>1.8).mean()),'| top ceiling bins (h, % of upper pts):',sorted(up))
res=0.02; xi=((P[:,0]-P[:,0].min())/res).astype(int); zi=((P[:,2]-P[:,2].min())/res).astype(int)
m=(H>0.3)&(H<1.8); g=np.zeros((zi.max()+1,xi.max()+1)); np.add.at(g,(zi[m],xi[m]),1)
g=np.log1p(g); Image.fromarray(255-(255*g/g.max()).astype(np.uint8)).save(f'{tag}_topdown.png')
