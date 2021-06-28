
[![GitHub repo size](https://img.shields.io/github/repo-size/marlonrcfranco/weather-guru)](https://github.com/marlonrcfranco/weather-guru)
[![GitHub top language](https://img.shields.io/github/languages/top/marlonrcfranco/weather-guru)](https://github.com/marlonrcfranco/weather-guru)
[![GitHub](https://img.shields.io/github/license/marlonrcfranco/weather-guru)](https://github.com/marlonrcfranco/weather-guru/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/marlonrcfranco/weather-guru?style=social)](https://github.com/marlonrcfranco/weather-guru/stargazers)

![Python](https://img.shields.io/badge/python%20-%2314354C.svg?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=flat-square&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=flat-square&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas%20-%23150458.svg?&style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=flat-square&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=flat-square&logo=Jupyter&logoColor=white)

# weather-guru
### Will it rain tomorrow?

<p align="center">
  <a href="https://colab.research.google.com/github/marlonrcfranco/weather-guru/blob/main/weather_guru.ipynb">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
</p>

## Intro 
### Goal 🌦
Implement an algorithm that performs **next day rain prediction** by training machine learning models on the target variable `RainTomorrow`.


### Dataset 📂
The dataset contains about **10 years of daily weather observations** from various locations in **Australia**.

**`RainTomorrow`** is the target variable to be  predicted. It means - it rained the next day, this column is `Yes` if the rain that day was 1mm or more.

## Data preprocessing 
### The raw data
Available at: [./data/weatherAUS.csv](https://github.com/marlonrcfranco/weather-guru/blob/main/data/weatherAUS.csv)

### Sample:

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99612</th>
      <td>2009-03-10</td>
      <td>MountGambier</td>
      <td>12.0</td>
      <td>27.6</td>
      <td>0.0</td>
      <td>4.2</td>
      <td>4.9</td>
      <td>E</td>
      <td>37.0</td>
      <td>ESE</td>
      <td>ESE</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>82.0</td>
      <td>36.0</td>
      <td>1020.5</td>
      <td>1017.5</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>16.5</td>
      <td>27.3</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>

### Columns:
```
RangeIndex: 145460 entries, 0 to 145459
Data columns (total 23 columns):
 #   Column         Non-Null Count   Dtype  
---  ------         --------------   -----  
 0   Date           145460 non-null  object 
 1   Location       145460 non-null  object 
 2   MinTemp        143975 non-null  float64
 3   MaxTemp        144199 non-null  float64
 4   Rainfall       142199 non-null  float64
 5   Evaporation    82670 non-null   float64
 6   Sunshine       75625 non-null   float64
 7   WindGustDir    135134 non-null  object 
 8   WindGustSpeed  135197 non-null  float64
 9   WindDir9am     134894 non-null  object 
 10  WindDir3pm     141232 non-null  object 
 11  WindSpeed9am   143693 non-null  float64
 12  WindSpeed3pm   142398 non-null  float64
 13  Humidity9am    142806 non-null  float64
 14  Humidity3pm    140953 non-null  float64
 15  Pressure9am    130395 non-null  float64
 16  Pressure3pm    130432 non-null  float64
 17  Cloud9am       89572 non-null   float64
 18  Cloud3pm       86102 non-null   float64
 19  Temp9am        143693 non-null  float64
 20  Temp3pm        141851 non-null  float64
 21  RainToday      142199 non-null  object 
 22  RainTomorrow   142193 non-null  object 
dtypes: float64(16), object(7)
```

### Convert wind Cardinal directions (string) to Degrees (float)
Wind directions are represented in Cardinal directions (e.g. N, S, SW,...). In order to make the dataset only with numbers, let's convert these directions into angles (degrees).

#### Before (row 0):
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
    </tr>
  </tbody>
</table>

<pre><span></span><span class="n">weather_df</span><span class="p">[</span><span class="s1">'WindGustDir'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="p">[</span><span class="s1">'WindGustDir'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">w</span><span class="p">:</span> <span class="n">portolan</span><span class="o">.</span><span class="n">middle</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">w</span><span class="p">))</span> <span class="k">if</span> <span class="nb">str</span><span class="p">(</span><span class="n">w</span><span class="p">)</span><span class="o">!=</span><span class="s1">'nan'</span> <span class="k">else</span> <span class="n">w</span><span class="p">)</span>
<span class="n">weather_df</span><span class="p">[</span><span class="s1">'WindDir9am'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="p">[</span><span class="s1">'WindDir9am'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">w</span><span class="p">:</span> <span class="n">portolan</span><span class="o">.</span><span class="n">middle</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">w</span><span class="p">))</span> <span class="k">if</span> <span class="nb">str</span><span class="p">(</span><span class="n">w</span><span class="p">)</span><span class="o">!=</span><span class="s1">'nan'</span> <span class="k">else</span> <span class="n">w</span><span class="p">)</span>
<span class="n">weather_df</span><span class="p">[</span><span class="s1">'WindDir3pm'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="p">[</span><span class="s1">'WindDir3pm'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">w</span><span class="p">:</span> <span class="n">portolan</span><span class="o">.</span><span class="n">middle</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">w</span><span class="p">))</span> <span class="k">if</span> <span class="nb">str</span><span class="p">(</span><span class="n">w</span><span class="p">)</span><span class="o">!=</span><span class="s1">'nan'</span> <span class="k">else</span> <span class="n">w</span><span class="p">)</span>
</pre>
#### After (row 0):
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>270.0</td>
      <td>270.0</td>
      <td>292.5</td>
    </tr>
  </tbody>
</table>

### Map the values 'Yes' and 'No' to 1 and 0
<div class=" highlight hl-python"><pre><span></span><span class="n">weather_df</span><span class="o">.</span><span class="n">RainToday</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">RainToday</span><span class="o">.</span><span class="n">map</span><span class="p">(</span><span class="nb">dict</span><span class="p">(</span><span class="n">Yes</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">No</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
<span class="n">weather_df</span><span class="o">.</span><span class="n">RainTomorrow</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">RainTomorrow</span><span class="o">.</span><span class="n">map</span><span class="p">(</span><span class="nb">dict</span><span class="p">(</span><span class="n">Yes</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">No</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
</pre></div>

### Add YEAR, MONTH and DAY columns based on column 'Date'
In order to catch ciclic behaviours, we can feed the model with the day, month and year separately and allow it to perceive patterns related to seasons, for example.

<div class=" highlight hl-python"><pre><span></span><span class="n">weather_df</span><span class="p">[</span><span class="s1">'year'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">Date</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">year</span>
<span class="n">weather_df</span><span class="p">[</span><span class="s1">'month'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">Date</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">month</span>
<span class="n">weather_df</span><span class="p">[</span><span class="s1">'day'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">Date</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">day</span>

<span class="c1"># Drop the Date column</span>
<span class="c1"># It is now replaced by year, month and day columns</span>
<span class="n">weather_df</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">'Date'</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

### Add latitude and longitude columns based on column 'Location'
The 'Location' column contains `string` values with the name of the location where the data was colected.
In order to convert the locations into numbers, we can use the coordinates for each one. 
With this aproach, the column 'Location' becomes three new columns: **latitude**, **longitude** and **altitude** (all in decimal degrees).
Example: Albury -> (-36.0804766, 146.9162795, 0.0)

**We could also plot the map of all the locations in the dataset, with its respective rainfall per day:**
<p align="center">
  <img src="https://raw.githubusercontent.com/marlonrcfranco/weather-guru/main/img/aus_rain_observations.png">
</p>






`TODO: Describe here which techiniques were used and why.`
