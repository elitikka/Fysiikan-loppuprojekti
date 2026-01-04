import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
from scipy.signal import butter, filtfilt


urlLinear = "https://raw.githubusercontent.com/elitikka/Fysiikan-loppuprojekti/main/Data/Linear%20Acceleration.csv"
urlLocation = "https://raw.githubusercontent.com/elitikka/Fysiikan-loppuprojekti/main/Data/Location.csv"

df = pd.read_csv(urlLinear)
loc = pd.read_csv(urlLocation)



# Laskelmat
# Alipäästösuodatin ja askelmäärä
def butter_lowpass_filter(data,cutoff,nyq,order):
    normal_cutoff = cutoff/nyq
    #Get the filter coefficents
    b,a=butter(order,normal_cutoff,btype='low', analog=False)
    y=filtfilt(b,a,data)
    return y

data=df['Linear Acceleration y (m/s^2)']
T_tot = df['Time (s)'].max()
n=len(df['Time (s)'])
fs = n/T_tot
nyq = fs/2
order=3
cutoff = 2

data_filt = butter_lowpass_filter(data,cutoff,nyq,order)

# Askelmäärä: 
jaksot = 0
for i in range(n-1):
    if data_filt[i]/data_filt[i+1] < 0:
        jaksot = jaksot + 1/2 

askelmaara = jaksot

# Fourier-analyysi

data_mean = np.mean(data)
data = data - data_mean

n = len(df['Time (s)'])
T_tot = df['Time (s)'].max()
dt = T_tot/n 

fourier = np.fft.fft(data, n)
psd = fourier*np.conj(fourier)/n
freq = np.fft.fftfreq(n,dt) 
L = np.arange(1, int(n/2))

f_max = freq[L][psd[L] == np.max(psd[L])][0]
T = 1/f_max
t = df['Time (s)']
steps = f_max*np.max(t)

# GPS-data
loc = loc[loc['Horizontal Accuracy (m)'] <10]
loc = loc.reset_index(drop=True)
lat1 = loc['Latitude (°)'].mean()
long1 = loc['Longitude (°)'].mean()

def haversine(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2 = map(radians, [lon1,lat1,lon2,lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin (dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # radius of earth in km
    return c * r

loc['Distance_calc'] = np.zeros(len(loc))
for i in range(len(loc)-1): 
    lon1 = loc['Longitude (°)'][i]
    lon2 = loc['Longitude (°)'][i+1]
    lat1 = loc['Latitude (°)'][i]
    lat2 = loc['Latitude (°)'][i+1]
    loc.loc[i+1, 'Distance_calc'] = haversine(lon1,lat1,lon2,lat2)

loc['total_distance'] = loc['Distance_calc'].cumsum()
total_distance = loc['total_distance'].iloc[-1]
avg_speed = loc['Velocity (m/s)'].mean()

keskinopeus = avg_speed
askelpituus = total_distance*1000/askelmaara # askelmäärä metreissä


# Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta
# Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella
# GPS: Keskinopeus
# GPS: Kuljettu matka
# Askelpituus (lasketun askelmäärän ja matkan perusteella)


# Print values:

st.title("Tulokset")

st.header("Lasketut tulokset")
st.subheader("Askelmäärät")
st.write("Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta: {jaksot:.0f}")
st.write("Askelmäärä laskettuna Fourier-analyysin perusteella: {steps:.0f}")

st.subheader("GPS tulokset")
st.write("Keskinopeus: {keskinopeus:.2f} m/s")
st.write("Kuljettu matka: {total_distance:.3f} km")

st.subheader("Askelpituus")
st.write("Askelpituus: {askelpituus: .2f} m")



st.header("Kuvaajat")

st.subheader("Suodatettu kiihtyvyysdatan y-komponentti")
fig1, ax1 = plt.subplots(figsize=(12,4))
ax1.plot(df['Time (s)'][:1000], data[:1000], label='data', alpha=0.7)
ax1.plot(df['Time (s)'][:1000], data_filt[:1000], label='suodatettu data', linewidth=2)
ax1.set_xlim(0,8)
ax1.set_ylim(-3,3)
ax1.set_xlabel('Aika [s]')
ax1.set_ylabel('Kiihtyvyys y [m/s²]')
ax1.legend()
ax1.grid()
st.pyplot(fig1)

st.subheader("Tehospektri")
fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(freq[L], psd[L].real)
ax2.set_xlabel('Taajuus [Hz]')
ax2.set_ylabel('Teho')
ax2.set_xlim(0,10)
ax2.grid()
st.pyplot(fig2)

st.subheader("Karttakuva")
lat1 = loc['Latitude (°)'].mean()
long1 = loc['Longitude (°)'].mean()
my_map = folium.Map(location=[lat1,long1], zoom_start=17)
folium.PolyLine(loc[['Latitude (°)', 'Longitude (°)']], color='red', weight=3).add_to(my_map)
st_folium(my_map, width=700, height=500)