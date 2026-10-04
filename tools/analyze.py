import cv2, numpy as np, json
V="/mnt/user-data/uploads/Woman_tracking_cursor_portrait_1080p_20261003235201.mp4"
cap=cv2.VideoCapture(V)
n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
W=int(cap.get(3)); H=int(cap.get(4))
print("frames",n,"fps",fps,"size",W,H,"dur",n/fps)
ok,f0=cap.read()
bg=np.median(f0[10:80,10:200].reshape(-1,3),axis=0)[::-1].astype(int)
print("bg RGB",bg,"hex #%02x%02x%02x"%tuple(bg))
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
cents=[];
i=0
while True:
    ok,f=cap.read()
    if not ok: break
    r=f[:,:,2].astype(int);g=f[:,:,1].astype(int);b=f[:,:,0].astype(int)
    m=(g>110)&(b<200)&(r>150)&((r-b)>40)&(g-b>5)
    m[700:,:]=False; m[:,:560]=False; m[:,1360:]=False
    ys,xs=np.nonzero(m)
    cents.append((xs.mean(),ys.mean(),len(xs)) if len(xs)>500 else (np.nan,np.nan,0))
    i+=1
c=np.array(cents)
np.save("cents.npy",c)
mx,my=np.nanmedian(c[-20:,0]),np.nanmedian(c[-20:,1])
print("neutral(end) centroid",mx,my)
for k in range(0,len(c),max(1,len(c)//60)):
    dx,dy=c[k,0]-mx,c[k,1]-my
    print(k,round(dx,1),round(dy,1),int(c[k,2]))
