import numpy as np, pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
o=pd.read_csv('odometry.csv',skipinitialspace=True)
pts=[]
s=256/1920
for i in range(0,len(o),20):
    r=o.iloc[i]; f=int(r.frame)
    d=np.array(Image.open(f'depth/{f:06d}.png')).astype(np.float32)/1000
    c=np.array(Image.open(f'confidence/{f:06d}.png'))
    v,u=np.nonzero((c==2)&(d>0.2)&(d<5))
    z=d[v,u]; fx,fy,cx,cy=r.fx*s,r.fy*s,r.cx*s,r.cy*s
    X=(u+0.5-cx)/fx*z; Y=(v+0.5-cy)/fy*z; Z=z
    P=np.stack([X,Y,Z],1)
    Rm=R.from_quat([r.qx,r.qy,r.qz,r.qw]).as_matrix()
    pts.append(P@Rm.T+np.array([r.x,r.y,r.z]))
P=np.concatenate(pts); np.save('pts.npy',P)
print('points',len(P))
h,e=np.histogram(P[:,1],bins=np.arange(P[:,1].min(),P[:,1].max(),0.02))
top=np.argsort(h)[-6:]
print('height peaks (y, count):',sorted([(round(e[i]+0.01,2),h[i]) for i in top]))
