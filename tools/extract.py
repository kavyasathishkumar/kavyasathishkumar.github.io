import cv2, numpy as np, os, json, math
V="/mnt/user-data/uploads/Woman_tracking_cursor_portrait_1080p_20261003235201.mp4"
OUT="/home/claude/site/kavyasathishkumar.github.io-main/frames"; os.makedirs(OUT,exist_ok=True)
c=np.load("cents.npy"); n=len(c)
mx,my=np.nanmedian(c[-20:,0]),np.nanmedian(c[-20:,1])
dx,dy=c[:,0]-mx,c[:,1]-my
mag=np.hypot(dx,dy); th=np.degrees(np.arctan2(dy,dx))   # screen coords: right=0, down=90, left=180, up=-90
m=np.load("motion.npy")
names=["UP","UP-RIGHT","RIGHT","DOWN-RIGHT","DOWN","DOWN-LEFT","LEFT","UP-LEFT"]
anch=[44,74,108,128,136,178,204,220]
comp=dict(zip(names,anch)); print("compass frames",comp)
N=64; sel=[]
def seg(A,B,k=8):
    cum=np.cumsum(m[A+1:B+1]+0.05); cum=cum/cum[-1]
    out=[A]
    for j in range(1,k):
        out.append(A+1+int(np.searchsorted(cum,j/k)))
    for i in range(1,len(out)): out[i]=max(out[i],out[i-1]+1) if B-A>=k else out[i]
    return out
for a in range(7): sel+=seg(anch[a],anch[a+1])
sel+=[220,223,226,229,31,35,39,42]    # UP-LEFT -> near-centre -> UP gap sector
sel=[int(x) for x in sel]
print("selected",sel,len(sel))
# decode needed frames
need=sorted(set(sel+[n-1]+list(comp.values())))
cap=cv2.VideoCapture(V); frames={}; i=0
while True:
    ok,f=cap.read()
    if not ok: break
    if i in need: frames[i]=f
    i+=1
f0=frames[n-1]
# background colour: edges of crop region, several samples
def samp(f,x,y): return f[y:y+16,x:x+16].reshape(-1,3).mean(0)[::-1]
pts=[(10,10),(10,520),(10,1050),(1890,10),(1890,520),(1050,10),(1050,1050),(900,10),(300,500),(1600,500)]
cols=np.array([samp(f0,x,y) for x,y in pts]); print("bg samples\n",cols.round(0))
bg=np.median(cols,axis=0).round().astype(int); hexv="#%02x%02x%02x"%tuple(bg); print("BG",hexv)
X0,X1=330,1590   # crop around character (also drops the corner sparkle mark)
W=X1-X0; OW=900; OH=int(1080*OW/W)
def save(idx,name):
    c_=frames[idx][:, X0:X1]; c_=cv2.resize(c_,(OW,OH),interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT}/{name}.webp",c_,[cv2.IMWRITE_WEBP_QUALITY,82])
for k,idx in enumerate(sel): save(idx,"f%02d"%k)
save(n-1,"center")
meta=dict(bg=hexv,count=N,w=OW,h=OH,face=dict(x=(988.5-X0)/W,y=462.0/1080),sel=sel,compass=comp)
json.dump(meta,open(f"{OUT}/meta.json","w"))
print("sizes KB:",sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT))//1024)
# contact sheet of compass + center
tiles=[cv2.resize(frames[comp[nm]][:,X0:X1],(300,252)) for nm in names]+[cv2.resize(frames[n-1][:,X0:X1],(300,252))]
for t,nm in zip(tiles,names+["CENTER"]): cv2.putText(t,nm,(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
sheet=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:6]),np.hstack(tiles[6:9])])
cv2.imwrite("/home/claude/sheet.jpg",sheet)
