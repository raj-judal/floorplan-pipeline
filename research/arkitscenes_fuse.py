import numpy as np, glob, os
from PIL import Image
from scipy.spatial.transform import Rotation as R
traj=np.loadtxt('lowres_wide.traj')
Rcw=R.from_rotvec(traj[:,1:4]).as_matrix(); tcw=traj[:,4:7]
Rwc=np.transpose(Rcw,(0,2,1)); C=-np.einsum('nij,nj->ni',Rwc,tcw)
intr={float(os.path.basename(f).split('_')[1][:-7]):f for f in glob.glob('lowres_wide_intrinsics/*.pincam')}
ik=np.array(sorted(intr))
def cloud(flip, step=1):
    out=[]; used=0
    for f in sorted(glob.glob('lowres_depth/*.png'))[::step]:
        ts=float(os.path.basename(f).split('_')[1][:-4])
        j=np.argmin(abs(traj[:,0]-ts))
        if abs(traj[j,0]-ts)>0.02: continue
        w,h,fx,fy,cx,cy=np.loadtxt(intr[ik[np.argmin(abs(ik-ts))]])
        d=np.array(Image.open(f)).astype(np.float32)/1000
        c=np.array(Image.open(f.replace('lowres_depth','confidence')))
        v,u=np.nonzero((c==2)&(d>0.2)&(d<5)); z=d[v,u]
        P=np.stack([(u+0.5-cx)/fx*z,(v+0.5-cy)/fy*z,z],1)*flip
        out.append(P@Rwc[j].T+C[j]); used+=1
    return np.concatenate(out), used
if __name__=='__main__':
    for name,flip in [('opencv',np.array([1,1,1])),('arkit',np.array([1,-1,-1]))]:
        P,u=cloud(flip,step=5); h,_=np.histogram(P[:,2],bins=np.arange(P[:,2].min(),P[:,2].max(),0.01))
        print(name,'frames',u,'max z-bin share %',round(100*h.max()/h.sum(),2))
