import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
import plotly.express as px
import plotly.graph_objects as go


urlLinear = "https://raw.githubusercontent.com/elitikka/Fysiikan-loppuprojekti/main/Linear Acceleration.csv"
urlLocation = "https://raw.githubusercontent.com/elitikka/Fysiikan-loppuprojekti/main/Location.csv"

df = pd.read_csv(urlLinear)
loc = pd.read_csv(urlLocation)

st.title('Fysiikan loppuprojekti')