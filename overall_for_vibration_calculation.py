import numpy as np
from matplotlib.widgets import Cursor
import array
import pandas as pd
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from pandas import DataFrame



########DATA READ from excel###############
##df1 = pd.read_excel('C:\\Companies\\Novosim\\NewProjects\\2019\\OTAM\\Titresim_Testi\\comp_vib_dc.xlsx')
##df2 = pd.read_excel('C:\\Companies\\Novosim\\NewProjects\\2019\\OTAM\\Titresim_Testi\\base_test_5_4000.xlsx')
###df3 = pd.read_excel('C:\\Companies\\Novosim\\NewProjects\\2019\\OTAM\\Titresim_Testi\\test_3_3173.xlsx')
##time=df1['Time']
##acc_z_1=df1['Accelerometer Z']

######DATA READ from csv###############
df1 = pd.read_csv('C:\\Users\\GkhnKznn\\Documents\\EDM\\Spider_DSA\\FFT2\\Default Folder 1-21-2024 3-13-27 PM\\BlackBox\\24663680\\REC0008.csv',sep=',',skiprows=16
                  , names=['time','ch1','1','2'])
print(df1.head())
time=df1['time']
acc_z_1=df1['ch1']


#CALCULATION FOR DATA1
N=4000
sample_rate=4000
noverlap=N//2
step=N-noverlap
data=[]


acc_1=list(np.float_(acc_z_1))
#acc_1=pd.DataFrame(acc_1)

T=1/sample_rate
x=np.linspace(0,1/(2*T),int(N/2))

for i in range(0, len(acc_1)-N, step):
    data.append(np.sqrt((1/N)*(np.power((acc_1[i:i+N]),2).sum())))

totaldatapoints=len(range(0, len(acc_1)-N, step))
totaltime=max(time)-2*N/sample_rate
xf=np.linspace(0,totaltime,totaldatapoints)

dfx=pd.DataFrame(xf)


dfx.to_csv('C:\\Users\\GkhnKznn\\Documents\\EDM\\Spider_DSA\\FFT2\\Default Folder 1-21-2024 3-13-27 PM\\BlackBox\\24663680\\REC0007.csv',sep=',')


fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
leg=ax.legend()

ax.plot(xf,data,'-b',label='DC ACC')


ax.set_xlabel('Zaman (sn)')
ax.set_ylabel('Titreşim (g)')
##ax.set_title('Kompresör OA İvme Seviyesi Kıyaslaması');

ax.grid(True)
ax.legend(loc='upper left', frameon=True)
cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
plt.show()


