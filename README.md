
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

# Table of Contents

* [Intro](#intro)
  * [Goal](#goal)
  * [Dataset](#dataset)
* [Data preprocessing](#data_preprocessing)
  * [The raw data](#the_raw_data)
  * [\[c\] connectors](#c_connectors)
  * [\[l\] list](#c_list)
  * [\[r\] read](#c_read)
  * [\[w\] write](#c_write)
  * [\[d\] delete](#c_delete)


<a name="intro"/>
## Intro 
<a name="goal"/>
### Goal 🌦
Implement an algorithm that performs **next day rain prediction** by training machine learning models on the target variable `RainTomorrow`.

<a name="dataset"/>
### Dataset 📂
The dataset contains about **10 years of daily weather observations** from various locations in **Australia**.

**`RainTomorrow`** is the target variable to be  predicted. It means - it rained the next day, this column is `Yes` if the rain that day was 1mm or more.

<a name="data_preprocessing"/>
## Data preprocessing 
<a name="the_raw_data"/>
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

### Split the Date into year, month and day columns
In order to catch ciclic behaviours, we can feed the model with the day, month and year separately and allow it to perceive patterns related to seasons, for example.

<div class=" highlight hl-python"><pre><span></span><span class="n">weather_df</span><span class="p">[</span><span class="s1">'year'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">Date</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">year</span>
<span class="n">weather_df</span><span class="p">[</span><span class="s1">'month'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">Date</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">month</span>
<span class="n">weather_df</span><span class="p">[</span><span class="s1">'day'</span><span class="p">]</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">Date</span><span class="o">.</span><span class="n">dt</span><span class="o">.</span><span class="n">day</span>
</pre></div>

### Add latitude and longitude columns based on column 'Location'
The 'Location' column contains `string` values with the name of the location where the data was colected.
In order to convert the locations into numbers, we can use the coordinates for each one. 
With this aproach, the column 'Location' becomes three new columns: **latitude**, **longitude** and **altitude** (all in decimal degrees).

Example: location `Albury` becomes latitude `-36.0804766`, lngitude `146.9162795` and altitude `0.0`.

**We could also plot the map of all the locations in the dataset, with its respective rainfall per day:**
<p align="center">
  <img src="https://raw.githubusercontent.com/marlonrcfranco/weather-guru/main/img/aus_rain_observations.png">
</p>

## Data Cleaning
### Remove rows with null values in the target column
If there's no value in the target column, we cannot use it into our training or test set. So we can remove those rows from the dataset.

<pre>Before:
  Unique values in the RainTomorrow column:  &lt;IntegerArray&gt;
[0, 1, &lt;NA&gt;]
Length: 3, dtype: Int64
 Total number of rows: [ 145460 ]

After:
  Unique values in the RainTomorrow column:  &lt;IntegerArray&gt;
[0, 1]
Length: 2, dtype: Int64
 Total number of rows: [ 142193 ] ( 3267  rows removed )
</pre>

### Filling missing values
In this step, we use the 'mean' strategy to fill the missing values of the features.

<div class=" highlight hl-python"><pre><span></span><span class="n">imputer</span> <span class="o">=</span> <span class="n">SimpleImputer</span><span class="p">(</span><span class="n">missing_values</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">nan</span><span class="p">,</span> <span class="n">strategy</span><span class="o">=</span><span class="s1">'mean'</span><span class="p">)</span>
<span class="c1"># transform the dataset</span>
<span class="n">weather_df_transformed</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">imputer</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">weather_df</span><span class="p">))</span>
<span class="n">weather_df_transformed</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">columns</span>
<span class="n">weather_df_transformed</span><span class="o">.</span><span class="n">index</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">index</span>

<span class="n">weather_df</span> <span class="o">=</span> <span class="n">weather_df_transformed</span>
</pre></div>


### Correlation

<p align="center">
  <img src="https://raw.githubusercontent.com/marlonrcfranco/weather-guru/main/img/correlation_matrix.png">
</p>

There are some correlation in the dataset. Enough to proceed with the work.

## Feature engineering
### Normalize features
The MinMaxScaler puts the values between 0 and 1, which further improves the model performance.
<div class=" highlight hl-python"><pre><span></span><span class="n">scaler</span> <span class="o">=</span> <span class="n">MinMaxScaler</span><span class="p">(</span><span class="n">feature_range</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span>
<span class="n">values</span> <span class="o">=</span> <span class="n">scaler</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">values</span><span class="p">)</span>
</pre></div>

### Split into input and outputs
X = features

Y = target

`TARGET_NAME = 'RainTomorrow'`

<div class=" highlight hl-python"><pre><span></span><span class="n">target_col_idx</span> <span class="o">=</span> <span class="n">weather_df</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">get_loc</span><span class="p">(</span><span class="n">TARGET_NAME</span><span class="p">)</span>

<span class="n">X</span> <span class="o">=</span> <span class="n">values</span><span class="p">[:,</span><span class="mi">0</span><span class="p">:</span><span class="n">target_col_idx</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">float</span><span class="p">)</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">values</span><span class="p">[:,</span><span class="n">target_col_idx</span><span class="p">]</span>
</pre></div>

### Encode class values as integers
<div class=" highlight hl-python"><pre><span></span><span class="n">encoder</span> <span class="o">=</span> <span class="n">LabelEncoder</span><span class="p">()</span>
<span class="n">encoder</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">Y</span><span class="p">)</span>
<span class="n">encoded_Y</span> <span class="o">=</span> <span class="n">encoder</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">Y</span><span class="p">)</span>
</pre></div>

## Design the model
We want to predict a boolean value ('Yes' or 'No') for the target variable RainTomorrow. In this case, we need to use a **classification model**, istead of a **regression model**, which is used to predict real-world values (e.g. Rainfall).

<div class=" highlight hl-python"><pre><span></span><span class="k">def</span> <span class="nf">create_model</span><span class="p">():</span>
  <span class="n">model</span> <span class="o">=</span> <span class="n">Sequential</span><span class="p">(</span><span class="n">name</span><span class="o">=</span><span class="s1">'weather_guru'</span><span class="p">)</span>
  <span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="n">NEURONS_1</span><span class="p">,</span> <span class="n">input_dim</span><span class="o">=</span><span class="n">NUM_FEATURES</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'dense_1'</span><span class="p">))</span>
  <span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="n">NEURONS_2</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'dense_2'</span><span class="p">))</span>
  <span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'sigmoid'</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'output'</span><span class="p">))</span>
  <span class="n">adam_optmz</span> <span class="o">=</span> <span class="n">Adam</span><span class="p">(</span><span class="n">learning_rate</span><span class="o">=</span><span class="n">LEARNING_RATE</span><span class="p">)</span>
  <span class="c1"># Compile model</span>
  <span class="n">model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">loss</span><span class="o">=</span><span class="n">LOSS_FUNCTION</span><span class="p">,</span> <span class="n">optimizer</span><span class="o">=</span><span class="n">adam_optmz</span><span class="p">,</span> <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">'accuracy'</span><span class="p">])</span>
  <span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">summary</span><span class="p">())</span>
  <span class="k">return</span> <span class="n">model</span>

<span class="n">estimators</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">estimators</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="s1">'standardize'</span><span class="p">,</span> <span class="n">StandardScaler</span><span class="p">()))</span>
<span class="n">estimators</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="s1">'mlp'</span><span class="p">,</span> <span class="n">KerasClassifier</span><span class="p">(</span><span class="n">build_fn</span> <span class="o">=</span> <span class="n">create_model</span><span class="p">,</span> <span class="n">epochs</span> <span class="o">=</span> <span class="n">EPOCHS</span><span class="p">,</span> <span class="n">batch_size</span> <span class="o">=</span> <span class="n">BATCH_SIZE</span><span class="p">,</span> <span class="n">verbose</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)))</span>
<span class="n">pipeline</span> <span class="o">=</span> <span class="n">Pipeline</span><span class="p">(</span><span class="n">estimators</span><span class="p">)</span>
</pre></div>

#### EPOCHS = 100
The number of epochs is the number of complete passes through the training dataset.
The number 100 is arbitrary, chosen to improve the model accuracy and training time.
#### BATCH_SIZE = 1023
The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.
The number 1023 is also arbitrary, chosen to improve the model accuracy and training time.
#### K_FOLD_SPLITS = 5
The number of iterations used in the cross validation.
The number 5 is arbitrary, chosen to reduce the time the validation takes. 
#### LOSS_FUNCTION = 'binary_crossentropy'
Binary cross entropy compares each of the predicted probabilities to actual class output which can be either 0 or 1.
The loss function 'binary_crossentropy' was chosen since the target variable is binary.
#### LEARNING_RATE = 0.001
Learning rate used in the Adam optimizer.
The value of 0.001 is arbitrary, chosen by trial and error, evaluating the model's improvements.
#### NEURONS_1 = 27 
The number of neurons in the first hidden layer. 
The values 27 is arbitrary, chosen by trial and error, evaluating the model's improvements.
#### NEURONS_2 = 7
The number of neurons in the first hidden layer. 
The values 7 was chosen to 'force' the model to choose the most relevant features


## Train the model and validate it
We use the cross validation score to measure the accuracy of our model.
<div class=" highlight hl-python"><pre><span></span><span class="n">kfold</span> <span class="o">=</span> <span class="n">StratifiedKFold</span><span class="p">(</span><span class="n">n_splits</span><span class="o">=</span><span class="n">K_FOLD_SPLITS</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

<div class=" highlight hl-python"><pre><span></span><span class="o">%%</span><span class="n">time</span>
<span class="n">results</span> <span class="o">=</span> <span class="n">cross_val_score</span><span class="p">(</span><span class="n">pipeline</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">encoded_Y</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="n">kfold</span><span class="p">)</span>
</pre></div>

See details of the training in the notebook: 
<a href="https://colab.research.google.com/github/marlonrcfranco/weather-guru/blob/main/weather_guru.ipynb">
   <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
</a>

# Final Results
**'Accuracy: 85.73% (0.12%)'**

This is the final accuracy result of the model, after the Cross Validation Score.

If necessary, we can change the model topology or configurations to try better accuracy scores.

