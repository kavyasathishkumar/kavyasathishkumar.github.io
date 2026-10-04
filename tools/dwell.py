import cv2, numpy as np
V="/mnt/user-data/uploads/Woman_tracking_cursor_portrait_1080p_20261003235201.mp4"
cap=cv2.VideoCapture(V); g=[]
while True:
    ok,f=cap.read()
    if not ok: break
    g.append(cv2.resize(cv2.cvtColor(f[40:820,520:1400],cv2.COLOR_BGR2GRAY),(220,195)).astype(np.float32))
m=np.array([0]+[np.abs(g[i]-g[i-1]).mean() for i in range(1,len(g))])
s=np.convolve(m,np.ones(5)/5,mode='same'); np.save("motion.npy",s)
print("max",s.max().round(2))
print("".join("%d"%min(9,int(v*2)) for v in s))   # bar digits per frame
mins=[i for i in range(3,len(s)-3) if s[i]==s[i-3:i+4].min() and s[i]<0.55*np.median(s)+0.2]
print("minima",[(i,round(float(s[i]),2)) for i in mins])
