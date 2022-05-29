#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


#URL：https://www.kaggle.com/datasets/gagandeep16/car-sales/versions/1
#URL：https://www.kaggle.com/datasets/shiyinwang/car-sales2
#URL：https://www.kaggle.com/datasets/shiyinwang/sales-by-country
#URL：https://www.kaggle.com/datasets/mysarahmadbhat/toyota-used-car-listing
#URL：https://www.kaggle.com/datasets/shiyinwang/us-car-sales
#URL:https://www.kaggle.com/datasets/shiyinwang/year22


# ## **1. Data Collection and Handling**

# In[4]:


from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)


# In[6]:


car_df = pd.read_csv('/Users/manishrajmr/Downloads/Car_sales.csv.xls')
car_df


# In[7]:


car_df.dropna(subset=['Price_in_thousands','Engine_size', 'Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency','__year_resale_value'],inplace=True)


# In[8]:


car_df.sort_values(by=['Sales_in_thousands', 'Manufacturer'], ascending=True)
car_df


# ## **2. Descriptive Statistics**

# ## 2.1 Measures of Location and Variability

# In[9]:


car_df.loc[(car_df['Sales_in_thousands'] > 200)]


# In[10]:


car_df.describe(include="all")


# In[11]:


car_df['Resale Value'] = (car_df['__year_resale_value'])/car_df['Price_in_thousands']
car_df


# In[12]:


df_style = car_df.style.format({'Resale Value':'{:.2%}'})
display(df_style)


# In[13]:


car_df.loc[(car_df['Resale Value'] > 0.9),]


# In[14]:


car_df['Resale Value'].value_counts(sort=False, normalize=False, bins=5)


# In[15]:


car_df['Sales_in_thousands'].value_counts(sort=False, normalize=False, bins=5)


# In[16]:


car_df['Sales_in_thousands'].value_counts(sort=False, normalize=True, bins=5)


# In[17]:


from scipy import stats

stats.gmean(car_df['Sales_in_thousands'], axis=0)


# In[18]:


car_df['Sales_in_thousands'].max() - car_df['Sales_in_thousands'].min()


# In[19]:


car_df.agg(
    {
        'Sales_in_thousands': ['sum', 'mad', 'median', 'var', 'skew', 'kurt'],
        'Price_in_thousands': ['sum', 'mad', 'median', 'var', 'skew', 'kurt']
    }
)


# ## 2.2 Distributions

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')

car_df.hist(column=['Sales_in_thousands'], bins=5, cumulative=False, figsize=(8,6))


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')

car_df.hist(column=['Price_in_thousands'], bins=5, cumulative=False, figsize=(8,6))


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the size of the plot
plt.figure(figsize=(6,8))

# Create an outlier in the data column
car_df.loc[19,'Sales_in_thousands'] = 40

# Create a box plot for the automobile sales on March 2010
sns.boxplot(data=car_df, y='Sales_in_thousands', orient='v')


# In[23]:


plt.figure(figsize=(10,8))
sns.boxplot(data=car_df, y='Sales_in_thousands', x='Manufacturer', orient='v')
plt.xticks(rotation=45)


# In[24]:


car_df.loc[19,'Price_in_thousands'] = 0
sns.scatterplot(data=car_df, x='Price_in_thousands', y='Sales_in_thousands')


# ## 2.3 Variable Association

# In[25]:


car_df.loc[:,['Price_in_thousands','Sales_in_thousands']].cov()


# In[26]:


car_var = car_df

car_var = pd.DataFrame()

car_var['Sales'] = car_df['Sales_in_thousands'].astype(float)

car_var[['Price','Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency','Resale Value']] = car_df[['Price_in_thousands',
       'Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency','Resale Value']].astype(float)
car_var


# In[27]:


car_var.corr()


# In[28]:


plt.figure(figsize = (30, 25))
sns.heatmap(car_var.corr(), annot = True, cmap="YlGnBu")
plt.show()


# ## **3. Data Visualization**

# ## 3.1 Line Chart

# In[29]:


bins = car_df['Sales_in_thousands'].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()

bins[0] = 0.0
bins


# In[30]:


car_df['Sales_Cat'] = pd.cut(x=car_df['Sales_in_thousands'], bins=bins, labels=['0.0-16.767', '16.767-32.299', '32.299-73.203','73.203-540.561'])
car_df


# In[31]:


car_crosstab = pd.pivot_table(car_df, values='Model', index=['Manufacturer'], columns=['Sales_Cat'], aggfunc='count', fill_value=0)
car_crosstab


# In[32]:


car_crosstab.sort_values(by=['73.203-540.561'], ascending=False).iloc[:20]


# In[33]:


car_df['Sales_Timestamp'] = pd.to_datetime(car_df['Latest_Launch'], format='%m/%d/%Y')
car_df


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

# Create a line chart to visualize how startup valuation changes over time
sns.lineplot(data=car_df, x="Sales_Timestamp", y="Sales_in_thousands")


# In[35]:


plot_data = car_df.loc[(car_df['Manufacturer'].isin(['Ford', 'Toyota','Dodge'])),]

plt.figure(figsize=(12,6))

sns.lineplot(data=plot_data, x="Sales_Timestamp", y="Sales_in_thousands", hue='Manufacturer')


# In[36]:


plot_data = car_df.loc[(car_df['Vehicle_type'].isin(['Passenger', 'Car'])),]

plt.figure(figsize=(12,6))

sns.lineplot(data=plot_data, x="Sales_Timestamp", y="Sales_in_thousands", hue='Vehicle_type')


# In[37]:


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_items', 3)

def sparkline(data, figsize=(4, 0.25), **kwargs):
    """
    creates a sparkline
    """
    from matplotlib import pyplot as plt
    import base64
    from io import BytesIO

    data = list(data)

    *_, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    ax.plot(data)
    ax.fill_between(range(len(data)), data, len(data)*[min(data)], alpha=0.1)
    ax.set_axis_off()

    img = BytesIO()
    plt.savefig(img)
    plt.close()
    return '<img src="data:image/png;base64, {}" />'.format(base64.b64encode(img.getvalue()).decode())


# ## 3.2 Sparkline

# In[38]:


sparkline_crosstab = pd.pivot_table(car_df, values='Sales_in_thousands', index=['Manufacturer'], columns=['Sales_Timestamp'], aggfunc=np.sum, fill_value=0)
sparkline_crosstab


# In[39]:


from IPython.display import HTML

sparkline_crosstab['Sparkline'] = sparkline_crosstab.apply(sparkline, axis=1)

HTML(sparkline_crosstab[['Sparkline']].iloc[:20].to_html(escape=False))


# ## 3.3 Bubble Chart

# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

ax = sns.scatterplot(data=car_df, x='Sales_Timestamp', y='Manufacturer', size='Sales_in_thousands', sizes=(10, 2000), legend=False)

plt.show()


# ## 3.4 Bar Chart

# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))


ax = sns.barplot(x='Manufacturer', y='Sales_in_thousands', data=car_df)

ax.bar_label(ax.containers[0])

plt.xticks(rotation=45)

plt.show()


# ## 3.5 Treemap

# In[42]:


import plotly.express as px

car_df['Year'] = car_df['Sales_Timestamp'].dt.year
fig = px.treemap(data_frame=car_df,        
                 path=['Manufacturer', 'Vehicle_type', 'Model'], 
                 values='Sales_in_thousands',           
                 color='Year',                  
                 color_continuous_scale='Blues', 
                 width=1200, height=600)        
fig.show()


# ## 3.6 GIS Chart

# In[44]:


country_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/2067972/3431605/Country.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220529T173544Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=16957c895ee31a40c1830d98b110653690523553f74f0a8b79d97a56f5ddaebd1a2a163746eb6d0e82af2d8e02700ba05e92713d008dfbfceeb33e4e31704d9c89ad1aa78c17ae5e159222a0f97a3dc2923e93fcaaa55a40ebf8e3ad46a9b5a96a71367a5b9f1be2c82093dc045d9023d90b35968575167a06a8d69f0c84c711c0a013f03ad1327a3aae37d4c034dbced968b869229ba36ffa95bf48dc156a1b7aa82e7e3312b65554ff6c0e221b83a79a9a0cd8233e65a2e0e9c2da634659bc38615eb716287f956d7d5f0e18f58fb2353ec8b1552bfa79107809672aedbc4ebb8f74907e617f5d4e39e7611d0bbd6918f00d29bf7026e880a09175436fb013')
country_df


# In[64]:


brand_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/2071339/3438087/Sales%20by%20brand.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220529T180320Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=7134510490113086171890a2471faa4f37cfff13a4d48d3970a398571f8b2ec48fad1725ea71344eb533074c646d953dde5a3793deda5146d4c41d7a1dbbec8ea268d59aace56837e9244067d0fde7838fecafa39f222b17f871a2ea90ec1eb1559894ad5fa6bdacfcf98348d2b5ce5eeffc48f09fadab3a8d0f091ff40ecb654c86da23b19451303b832f54ee795ff8ae77971a7893403340f9b314c40b31bdc4d391cd3a527edd4f6861d729f7f39fa718046b372b7d42a44e966150a71df992db2c4f0d31fc32bf2dff6321a2ed0f31fc74f7b14bef83f09fc47cf6a2a95d2e9d538e0dba8f800a6df69f4fd8240ef10ff4f34bc7f6c2264d8e91ef2081f1')
brand_df.columns.name = 'Year'
brand_df


# In[65]:


brand_df['Growth Rate']=(brand_df['2021']-brand_df['2020'])/brand_df['2020']
brand_df = brand_df.style.format({'Growth Rate':'{:.2%}'})
brand_df


# In[67]:


year_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/2071339/3438087/Sales%20by%20year.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220529T180402Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=2c539ce57ef80973f1b84fb96708c8575e088eabe58d64eb0277cabdac3b29a937775c32c9a96e906ef0266a1f4fb88ca540e40165c578b0e2fe6f0c0636c4068ec2a0ec9c53c166127521b70f2f6c6b39f0135e80dd7cbbc324a0f8bb65887ddea7dcaed36e70fc0fe96c51a3f100c032697fb867b1d8215796621be376a0a6ec975affb6f5c11d922dbce2a6404bc3f52af52b62b930e5c0df4a0a3f7fdfadeaf78f4f394560ff008f11d7f871a6675dc6e301a8956f482d50e3ad67d419656f051abc3dda6325052c092ffc8e83ea771848c6dea945c9be6ec3d0242c0ea1bbfd2759bb9159bb73c1ddfbc3cc328223f533f67256d790dd133c7ae430c302')
year_df.columns.name = 'Year'
year_df


# In[68]:


year_df = pd.melt(year_df,id_vars="Region", var_name="Year",value_name="Car Sales")
year_df = year_df.sort_values(by = 'Year')
year_df


# In[69]:


import plotly.express as px
import pandas as pd
import numpy as np


fig_line=px.line(        
        year_df,
        x = "Year", #Columns from the data frame
        y = "Car Sales",
    color='Region',
        title = "Car Sales over Time"
)

    
fig_line.update_xaxes(tickangle= 45)  
fig_line.show()


# In[70]:


import plotly.graph_objects as go

fig_bar = go.Figure(go.Bar(
            x=[9615157, 4136018, 4065014, 3890981, 2777056, 2521525, 2054962, 1680512, 1287548, 1086100, 936172, 878200, 698693, 470500, 301915],
            y=['Toyota','Honda','Nissan','Hyundai','Kia','BMW','Mercedes-Benz','Audi','Mazda','Volkwagen','Tesla','Skoda','Volve','Sear','Porsche'],
            text=[9.62, 4.14, 4.07, 3.89, 2.78, 2.52, 2.05, 1.68, 1.29, 1.09, 9.36, 8.78, 6.99, 4.71, 3.02],
            orientation='h'))

fig_bar.update_layout(
    title={'text': 'Car Sales by Brand',
           'y':0.9,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'})

fig_bar.show()


# ## 3.7 Multiple Chart

# In[71]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Create figure with secondary y-axis
fig.add_trace(
    go.Bar(
        x=['Toyota','Honda','Nissan','Hyundai','Kia','BMW','Mercedes-Benz','Audi','Mazda','Volkwagen','Tesla','Skoda','Volve','Sear','Porsche'],
        y=[8692168, 4384179, 4029185, 3744737, 2606832, 2324809, 2164187, 1692773, 1243005, 1146400, 499535, 1004800, 661713, 426600, 272162],
        name='Car Sales 2020'),
        secondary_y=False,
)

fig.add_trace(
    go.Bar(
        x=['Toyota','Honda','Nissan','Hyundai','Kia','BMW','Mercedes-Benz','Audi','Mazda','Volkwagen','Tesla','Skoda','Volve','Sear','Porsche'],
        y=[9615157, 4136018, 4065014, 3890981, 2777056, 2521525, 2054962, 1680512, 1287548, 1086100, 936172, 878200, 698693, 470500, 301915],
        name='Car Sales 2021'),
        secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=['Toyota','Honda','Nissan','Hyundai','Kia','BMW','Mercedes-Benz','Audi','Mazda','Volkwagen','Tesla','Skoda','Volve','Sear','Porsche'],
        y=[10.62,-5.66,0.89,3.91,6.53,8.46,-5.05,0.72,3.58,5.26,87.41,-12.6,5.59,10.29,10.93],
        name='Growth Rate (%)'),
        secondary_y=True,
)

fig.update_layout(
    title={'text': 'Car Sales & Growth Rate 2021 by Countries', 
           'y':0.9,
           'x':0.43,
           'xanchor': 'center',
           'yanchor': 'top'})

fig.show()


# ## 3.8 Dashboard

# ## 3.9 Stacked Bar Chart

# In[73]:


year2_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/2078758/3451400/year2.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220529T180521Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=33ead285d93e0cc85b3342777f1141d1b784cf22395304a0eef10aefe70b94e08442888871d8bb063275e083b398f54d976e619ef339e62f9fda48ac032587ef7203ba7c49249ee1f889bfbdccb0b6e847ff53937aac6d0d345c2cad2d55685729f1015ee45bda4aa8b52ee2926bf3f9f2093a8377b81fe382602aa32b6ccfaba3715c1d7eb89c3ed72beb83e0e1df4369187d4fae9310b1f33d6fc274e017d263416867bdbe131b57cae4f7a76fa39662529b721693ebb2ab1b8299392475287b65b1eaaedcce0c9c3d5327a3705b9935c0eeaced763ed708f44fe426529e9221b7dae887519623db0e88178d336ad27f84110e8e8c0669e91266bd68f7eaa9')
year2_df.columns.name = 'Year'
year2_df


# In[74]:


year2_df = pd.melt(year2_df,id_vars="Region", var_name="Year",value_name="Car Sales")
year2_df


# In[75]:


year2_crosstab = pd.pivot_table(year2_df, values='Car Sales', index=['Year'], columns=['Region'])
year2_crosstab


# In[76]:


import matplotlib.pyplot as plt
year2_crosstab.plot(kind='bar', 
                    stacked=True, 
                    colormap='Set3', 
                    figsize=(14, 6))

plt.legend(loc="upper left", ncol=7)
plt.xlabel("Year")
plt.ylabel("Proportion")
plt.title("Car Sales by Key Market")
plt.xticks(rotation=360)

for n, x in enumerate([*year2_crosstab.index.values]):
    for (proportion, y_loc) in zip(year2_crosstab.loc[x],
                                   year2_crosstab.loc[x].cumsum()):
                
        plt.text(x=n - 0.17,
                 y=(y_loc - proportion) + (proportion / 2),
                 s=f'{np.round(proportion * 100, 1)}%', 
                 color="black",
                 fontsize=8,
                 fontweight="bold")


plt.show()


# ## 3.10 Pie Chart

# In[77]:


car_df['Vehicle_type'].value_counts()


# In[78]:


import plotly.graph_objects as go

labels = ['Passenger','Car']
values = [88, 29]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0])])
fig.update_layout(
    title={'text': 'Vehicle Type', 
           'y':0.9,
           'x':0.47,
           'xanchor': 'center',
           'yanchor': 'top'})
fig.show()


# In[79]:


car_df['Engine_size'].value_counts(bins=4)


# In[59]:


import plotly.graph_objects as go

labels = ['1-2.75','2.75-4.5','4.5-6.25','6.25-8.0']
values = [51, 53, 12, 1]

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0, 0])])
fig.update_layout(
    title={'text': 'Engine Size', 
           'y':0.9,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'})
fig.show()


# In[80]:


car_df['Resale Value'].value_counts(bins=4)


# In[81]:


import plotly.graph_objects as go

labels = ['47.8%-60.8%','60.8%-73.7%','73.7%-86.7%','86.7%-99.6%']
values = [33, 46, 31, 7]

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0, 0])])
fig.update_layout(
    title={'text': 'Resale Value', 
           'y':0.9,
           'x':0.47,
           'xanchor': 'center',
           'yanchor': 'top'})
fig.show()


# In[82]:


car_df['Horsepower'].value_counts(bins=4)


# In[83]:


import plotly.graph_objects as go

labels = ['55-154','154-253','253-351','351-450']
values = [44, 61, 11, 1]

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0, 0])])
fig.update_layout(
    title={'text': 'Horsepower', 
           'y':0.9,
           'x':0.5,
           'xanchor': 'center',
           'yanchor': 'top'})
fig.show()


# ## **4. Hypothesis Testing**

# In[84]:


toyota_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/1606014/2641119/toyota.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220529T180630Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=5a441228c2c2ba006fe6cbe680820f678a3f7f83deb34a7dcba4e2ff00d3d1f44aedd664f5db9ae765baf958658a5f42521cc53f5fa07df2b3159156c50e4e1ee442f3a69892a87abd101f5f4051f58874873a9952c5de54835ab6d45d4a30ee00660e95525c13cb3c76aa32b83caaeb3894de95cb2acd9c6ecdbec6f114333fba3c7ab14bd0d799e09e081da545398dd88f3c77a95298eec8bb45462cf7f6cba2685bbbf5050f1192c20c0fe2d7d25a6b63a7e44cfe68607997c204d511823cfbf9e2fec270f74ca0c540ab3c67ff7c1bb8a5cd916441ef354bd2ab54254654f03effa6cc8865bc6e6039f80d9a016320f0154b7cf8ca8111e3d604d47c41f0')
toyota_df


# In[85]:


toyota_df=toyota_df.loc[(toyota_df['year']==2020)]
toyota_df


# In[86]:


mean_list = [toyota_df.sample(frac=0.1, replace=False, random_state=seed)['price'].mean() for seed in range(500)]

# Convert the list of sample means to a data frame
mean_df = pd.DataFrame({'Sample Mean':mean_list})
mean_df


# In[87]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

# Use the seaborn.histplot() method to plot the frequency distribution of sample means
# Set kde to True to compute the kernel density estimate to smooth the distribution 
sns.histplot(data=mean_df, x='Sample Mean', kde=True)


# In[88]:


std_list = [toyota_df.sample(frac=0.1, replace=False, random_state=seed)['price'].std() for seed in range(500)]

# Convert the list of sample standard deviations (std) to a data frame
std_df = pd.DataFrame({'Sample Std':std_list})
std_df


# In[89]:


plt.figure(figsize=(12,6))

# Use the seaborn.histplot() method to plot the frequency distribution of sample standard deviation
sns.histplot(data=std_df, x='Sample Std', kde=True)


# In[90]:


plt.figure(figsize=(12,6))

# Use the seaborn.histplot() method to plot the frequency distribution of sample proportion
sns.histplot(data=toyota_df, x='price', kde=True)


# In[91]:


import math
from scipy.stats import t

sample_df = toyota_df.sample(frac=0.1, replace=False, random_state=1)

# A confidence level of 95%
alpha = 1 - 0.95
sample_mean = sample_df['price'].mean()
sample_std = sample_df['price'].std()
n = sample_df.shape[0]
# The degree of freedom for a sample size of sample_df.shape[0]
dof = n - 1

t_one_tail = t.ppf(alpha/2, dof)
error_margin = t_one_tail * (sample_std / math.sqrt(n))
print('Sample Mean: '+str(sample_mean))
print('Interval Estimate of a Population Mean: ['+str(sample_mean-abs(error_margin))+', '+str(sample_mean+abs(error_margin))+']')


# In[92]:


# Calculate t_score by applying the t-statistic formula
t_score = (sample_mean - 34790) / (sample_std / math.sqrt(n))
# Upper-tailed t-test for the mean menu price of Michelin restaurants
p_value = t.sf(abs(t_score), n-1)
p_value


# ## **5. Regression Analysis**

# ## 5.1 Linear Regression

# In[93]:


endog_df = car_df[['Sales_in_thousands']]
endog_df


# In[94]:


import statsmodels.api as sm
exog_df = car_df[['Price_in_thousands']]

exog_df = sm.add_constant(exog_df, prepend=True)
exog_df


# In[95]:


pd.set_option('mode.chained_assignment', None)

endog_df['Sales_in_thousands_log'] = np.log(endog_df['Sales_in_thousands']+1)
endog_df


# In[96]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

sns.histplot(data=endog_df, x='Sales_in_thousands', kde=True, ax=axs[0])
sns.histplot(data=endog_df, x='Sales_in_thousands_log', kde=True, ax=axs[1])

# Display normality test results as the plot title
axs[0].set_title('Normality test for Sales_in_thousands:' +
      '\n'+ str(stats.normaltest(endog_df['Sales_in_thousands'])) +
      '\nSkewness: ' + str(stats.skew(endog_df['Sales_in_thousands'])) +
      '\nKurtosis: ' + str(stats.kurtosis(endog_df['Sales_in_thousands']) + 3))
axs[1].set_title('Normality test for log(Sales_in_thousands):' +
      '\n'+ str(stats.normaltest(endog_df['Sales_in_thousands_log'])) +
      '\n' + str(stats.jarque_bera(endog_df['Sales_in_thousands_log'])) +
      '\nSkewness: ' + str(stats.skew(endog_df['Sales_in_thousands_log'])) +
      '\nKurtosis: ' + str(stats.kurtosis(endog_df['Sales_in_thousands_log']) + 3))

fig.show()


# In[97]:


import statsmodels.api as sm

mod_log = sm.OLS(endog_df['Sales_in_thousands_log'], exog_df)

res_log = mod_log.fit()
print(res_log.summary())


# In[98]:


mod = sm.OLS(endog_df['Sales_in_thousands'], exog_df)
res = mod.fit()
print(res.summary())


# In[99]:


import statsmodels.stats.stattools as sss

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

sns.histplot(data=res.resid, kde=True, ax=axs[0])
sns.histplot(data=res_log.resid, kde=True, ax=axs[1])

# Display normality test results as the plot title
axs[0].set_title('Test residual distribution of Sales_in_thousands:' +
      '\n'+ str(stats.normaltest(res.resid)) +
      '\n' + str(stats.jarque_bera(res.resid)) +
      '\nSkewness: ' + str(stats.skew(res.resid)) +
      '\nKurtosis: ' + str(stats.kurtosis(res.resid) + 3) +
      '\nAutocorrelation: ' + str(sss.durbin_watson(res.resid)))
axs[1].set_title('Test residual distribution of log(Sales_in_thousands):' +
      '\n'+ str(stats.normaltest(res_log.resid)) +
      '\n' + str(stats.jarque_bera(res_log.resid)) +
      '\nSkewness: ' + str(stats.skew(res_log.resid)) +
      '\nKurtosis: ' + str(stats.kurtosis(res_log.resid) + 3) +
      '\nAutocorrelation: ' + str(sss.durbin_watson(res_log.resid)))

fig.show()


# In[100]:


exog_df[['Engine_size', 'Horsepower','Length','Wheelbase','Width','Curb_weight','Fuel_capacity','Fuel_efficiency','Resale Value']] = car_df[['Engine_size', 'Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency','Resale Value']]
exog_df


# In[101]:


mod = sm.OLS(endog_df['Sales_in_thousands_log'], exog_df)
res_multi = mod.fit()
print(res_multi.summary())


# In[102]:


Vehicle_type_dummies= pd.get_dummies(car_df['Vehicle_type']).iloc[:,1:]
Vehicle_type_dummies


# In[103]:


exog_df[Vehicle_type_dummies.columns] = Vehicle_type_dummies
exog_df


# In[104]:


mod = sm.OLS(endog_df['Sales_in_thousands_log'], exog_df)
res_cat = mod.fit()
print(res_cat.summary())


# In[105]:


car_var = car_df

car_var = pd.DataFrame()

car_var['Sales'] = car_df['Sales_in_thousands'].astype(float)

car_var[['Price','Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency','Resale Value']] = car_df[['Price_in_thousands',
       'Engine_size','Horsepower','Wheelbase','Width','Length','Curb_weight','Fuel_capacity','Fuel_efficiency','Resale Value']].astype(float)
car_var


# ## 5.2 Quadratic Regression Models

# In[106]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(car_var)


# In[107]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))

sns.regplot(data=car_var, x='Price', y='Sales')


# In[108]:


import statsmodels.api as sm

quad_endog = car_var[['Sales']]
quad_exog = car_var[['Price']]
quad_exog = sm.add_constant(quad_exog, prepend=True)

quad_exog['Price^2'] = quad_exog['Price'] * quad_exog['Price']
quad_endog.head(), quad_exog.head()


# In[109]:


quad_mod = sm.OLS(quad_endog, quad_exog)
quad_res = quad_mod.fit()
print(quad_res.summary())


# In[110]:


quad_pred = quad_res.get_prediction().summary_frame()
# Rename all the parameters
quad_pred.columns = ['pred_Sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
# Add observed 'happiness' and 'obesity' to quad_pred from the sample 
quad_pred[['obs_Sales', 'Price']] = car_var[['Price', 'Sales']]
quad_pred.head()


# In[111]:


plt.figure(figsize=(16,8))

plot_data = quad_pred.copy()

plot_data = plot_data.sort_values(by='Price', ascending=True)

sns.lineplot(data=plot_data, x='Price', y='pred_Sales', color='red')

sns.scatterplot(data=plot_data, x='Price', y='obs_Sales', color='blue')

plt.fill_between(x=plot_data['Price'], y1=plot_data['ci_lower'], y2=plot_data['ci_upper'], color='teal', alpha=0.2)

plt.fill_between(x=plot_data['Price'], y1=plot_data['pi_lower'], y2=plot_data['pi_upper'], color='skyblue', alpha=0.2)


# In[112]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))

sns.regplot(data=car_var, x='Engine_size', y='Sales')


# In[113]:


import statsmodels.api as sm

quad_endog = car_var[['Sales']]
quad_exog = car_var[['Engine_size']]
quad_exog = sm.add_constant(quad_exog, prepend=True)

quad_exog['Engine_size^2'] = quad_exog['Engine_size'] * quad_exog['Engine_size']
quad_endog.head(), quad_exog.head()


# In[114]:


quad_pred = quad_res.get_prediction().summary_frame()
# Rename all the parameters
quad_pred.columns = ['pred_Sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
# Add observed 'happiness' and 'obesity' to quad_pred from the sample 
quad_pred[['obs_Sales', 'Engine_size']] = car_var[['Price', 'Engine_size']]
quad_pred.head()


# In[115]:


plt.figure(figsize=(16,8))

plot_data = quad_pred.copy()

plot_data = plot_data.sort_values(by='Engine_size', ascending=True)

sns.lineplot(data=plot_data, x='Engine_size', y='pred_Sales', color='red')

sns.scatterplot(data=plot_data, x='Engine_size', y='obs_Sales', color='blue')

plt.fill_between(x=plot_data['Engine_size'], y1=plot_data['ci_lower'], y2=plot_data['ci_upper'], color='teal', alpha=0.2)

plt.fill_between(x=plot_data['Engine_size'], y1=plot_data['pi_lower'], y2=plot_data['pi_upper'], color='skyblue', alpha=0.2)


# In[116]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))

sns.regplot(data=car_var, x='Fuel_efficiency', y='Sales')


# In[117]:


import statsmodels.api as sm

quad_endog = car_var[['Sales']]
quad_exog = car_var[['Fuel_efficiency']]
quad_exog = sm.add_constant(quad_exog, prepend=True)

quad_exog['Fuel_efficiency^2'] = quad_exog['Fuel_efficiency'] * quad_exog['Fuel_efficiency']
quad_endog.head(), quad_exog.head()


# In[118]:


quad_pred = quad_res.get_prediction().summary_frame()
# Rename all the parameters
quad_pred.columns = ['pred_Sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
# Add observed 'happiness' and 'obesity' to quad_pred from the sample 
quad_pred[['obs_Sales', 'Fuel_efficiency']] = car_var[['Price', 'Fuel_efficiency']]
quad_pred.head()


# In[119]:


plt.figure(figsize=(16,8))

plot_data = quad_pred.copy()

plot_data = plot_data.sort_values(by='Fuel_efficiency', ascending=True)

sns.lineplot(data=plot_data, x='Fuel_efficiency', y='pred_Sales', color='red')

sns.scatterplot(data=plot_data, x='Fuel_efficiency', y='obs_Sales', color='blue')

plt.fill_between(x=plot_data['Fuel_efficiency'], y1=plot_data['ci_lower'], y2=plot_data['ci_upper'], color='teal', alpha=0.2)

plt.fill_between(x=plot_data['Fuel_efficiency'], y1=plot_data['pi_lower'], y2=plot_data['pi_upper'], color='skyblue', alpha=0.2)


# ## 5.3 Piecewise Linear Regression Model

# In[120]:


import statsmodels.api as sm

pw_endog = car_var[['Sales']]
pw_exog = car_var[['Price']]
pw_exog = sm.add_constant(pw_exog, prepend=True)
pw_endog.head(), pw_exog.head()


# In[121]:


pw_exog['beyond_knot'] = (pw_exog['Price'] > 30).astype(int)

pw_exog['relative_Price'] = (pw_exog['Price'] - 30) * pw_exog['beyond_knot']
pw_exog.head()


# In[122]:


pw_mod = sm.OLS(pw_endog, pw_exog[['const','Price','relative_Price']])
pw_res = pw_mod.fit()
print(pw_res.summary())


# In[123]:


pw_pred = pw_res.get_prediction().summary_frame()
pw_pred.columns = ['pred_Sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
pw_pred[['obs_Sales', 'Price']] = car_var[['Sales', 'Price']]
pw_pred.head()


# In[124]:


plt.figure(figsize=(16,8))

plot_data = pw_pred.copy()

plot_data = plot_data.sort_values(by='Price', ascending=True)

sns.lineplot(data=plot_data, x='Price', y='pred_Sales', color='red')

sns.scatterplot(data=plot_data, x='Price', y='obs_Sales', color='blue')

plt.fill_between(x=plot_data['Price'], y1=plot_data['ci_lower'], y2=plot_data['ci_upper'], color='teal', alpha=0.2)

plt.fill_between(x=plot_data['Price'], y1=plot_data['pi_lower'], y2=plot_data['pi_upper'], color='skyblue', alpha=0.2)


# ## 5.4 Interaction between Independent Variables

# In[125]:


car_df[['Sales','Price','Engine_size ','Fuel_efficiency']] = car_df[['Sales_in_thousands','Price_in_thousands',
       'Engine_size','Fuel_efficiency']].astype(float)


# In[126]:


plt.figure(figsize=(16,8))
sns.scatterplot(data=car_df, x='Price', y='Sales', hue='Vehicle_type')


# In[127]:


import statsmodels.api as sm

inter_endog = car_df[['Sales']]
inter_exog = car_df[['Price']]
inter_exog = sm.add_constant(inter_exog, prepend=True)
inter_endog.head(), inter_exog.head()


# In[128]:


inter_exog['Vehicle_type'] = car_df['Vehicle_type'].astype('category').cat.codes

inter_exog['Price*Vehicle_type'] = inter_exog['Price'] * inter_exog['Vehicle_type']
inter_exog.head()


# In[129]:


inter_mod = sm.OLS(inter_endog['Sales'], inter_exog)
inter_res = inter_mod.fit()
print(inter_res.summary())


# In[130]:


inter_pred = inter_res.get_prediction().summary_frame()
inter_pred.columns = ['pred_Sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
inter_pred['obs_Sales'] = inter_endog['Sales']
inter_pred[['Price', 'Vehicle_type']] = inter_exog[['Price', 'Vehicle_type']]
inter_pred.head()


# In[131]:


plt.figure(figsize=(16,8))

plot_data = inter_pred.copy()

plot_data = plot_data.sort_values(by='Price', ascending=True)

passenger = plot_data.loc[(plot_data['Vehicle_type'] == 1),]
car = plot_data.loc[(plot_data['Vehicle_type'] == 0),]

sns.lineplot(data=passenger, x='Price', y='pred_Sales', color='blue')
sns.lineplot(data=car, x='Price', y='pred_Sales', color='red')

sns.scatterplot(data=passenger, x='Price', y='obs_Sales', color='skyblue')
sns.scatterplot(data=car, x='Price', y='obs_Sales', color='orange')

plt.fill_between(x=passenger['Price'], y1=passenger['ci_lower'], y2=passenger['ci_upper'], color='skyblue', alpha=0.2)
plt.fill_between(x=car['Price'], y1=car['ci_lower'], y2=car['ci_upper'], color='orange', alpha=0.2)

plt.fill_between(x=passenger['Price'], y1=passenger['pi_lower'], y2=passenger['pi_upper'], color='skyblue', alpha=0.2)
plt.fill_between(x=car['Price'], y1=car['pi_lower'], y2=car['pi_upper'], color='orange', alpha=0.2)


# ## **6. Forecasting**

# ## 6.1 Forecasting via Linear Regression

# In[133]:


usa_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/2077371/3448950/US%20Car%20Sales.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220529%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220529T180800Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=7c712c2321c020eacce74bbf9dc3831731c9416990dd53f6975005f2768d6d394b14e693c73da8b14b30c1993f23f3be70c26d1fd18d4c79394014eb742bc33ba88f80f35506d3af90c5b27083976a0d7e6f1d5e7eb0ee0d5df018ab597fd6d829c7858fd2b958187acb768c3adacc20aaa553b3f1cda25c8c2a1af43aec433bb81d8f0f17bfb3bff6aa5ece81e0ee0fd5753ceb86250e1d4a238ea648a71578033b7a13e0007483e4e97a71e36d2100dcd8e065993c52ab8dbddd85619bc9803517afd77e43c822f6470972b57233e28d30c1f18dcde94cbd035f3a14b27898c6162c5ace1b93713834fce3d244e98cff346d94d53385a6c021733b1d9aa6fc')
usa_df


# In[134]:


usa_df['timestamp'] = pd.to_datetime(usa_df['Month'], format="%Y-%m")
usa_df.columns


# In[135]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))


sns.lineplot(data=usa_df, x='timestamp', y='Sales', color='skyblue', alpha=0.5, label='Car Sales')


# In[136]:


def forecast_accuracy(df, obs, pred):
    '''
    This function calculates four types of forecast errors.
    
    Parameters
    ----------
    df : DataFrame
        Container of observations and predictions.
    obs : str
        Column index for the observations.
    pred : str
        Column index for the predictions.

    Returns
    -------
    str
        A summary of all four types of forecast errors.
    '''
    mfe = (df[obs] - df[pred]).mean()
    mae = abs(df[obs] - df[pred]).mean()
    mse = ((df[obs] - df[pred]) ** 2).mean()
    mape = (((df[obs] - df[pred]) / df[obs]) * 100).mean()

    return 'Mean Forcasting Error (MFE): '+str(mfe)+           '\nMean Absolute Error (MAE): '+str(mae)+           '\nMean Squared Error (MSE): '+str(mse)+           '\nMean Absolute Percentage Error (MAPE): '+str(mape)


# In[137]:


forecast_df = pd.DataFrame()
forecast_df[['timestamp', 'sales']] = usa_df[['timestamp', 'Sales']]
forecast_df.iloc[:10]


# In[138]:


plt.figure(figsize=(16,8))

sns.lineplot(data=forecast_df, x='timestamp', y='sales', color='skyblue', alpha=1)


# In[139]:


averages = [usa_df['Sales'][:i].mean() for i in range(1, usa_df.shape[0])]
averages.insert(0, None)
forecast_df['total_average'] = averages
forecast_df.iloc[:10]


# In[140]:


print(forecast_accuracy(forecast_df, 'sales','total_average'))


# In[141]:


plt.figure(figsize=(16,8))

sns.lineplot(data=forecast_df, x='timestamp', y='sales', color='skyblue', alpha=0.5)
sns.lineplot(data=forecast_df, x='timestamp', y='total_average', color='red', alpha=1)


# In[142]:


# Calculate moving averages for 'new_cases' with a 3-day rolling window
forecast_df['3_month_moving_averages'] = usa_df.rolling(window=3)['Sales'].mean().shift(1)
# Calculate moving averages for 'new_cases' with a 7-day rolling window
forecast_df['7_month_moving_averages'] = usa_df.rolling(window=7)['Sales'].mean().shift(1)
forecast_df.iloc[:10]


# In[143]:


print(forecast_accuracy(forecast_df, 'sales', '3_month_moving_averages'))


# In[144]:


print(forecast_accuracy(forecast_df, 'sales', '7_month_moving_averages'))


# In[145]:


plt.figure(figsize=(16,8))

sns.lineplot(data=forecast_df, x='timestamp', y='sales', color='skyblue', alpha=0.5)
sns.lineplot(data=forecast_df, x='timestamp', y='3_month_moving_averages', color='orange', alpha=0.5, label='3-Month Moving Averages')
sns.lineplot(data=forecast_df, x='timestamp', y='7_month_moving_averages', color='red', alpha=0.5, label='7-Month Moving Averages')


# In[146]:


# Calculate exponentially weighted averages with a smoothing constant 0.2 for the 'new_cases' column
forecast_df['0.2_exponential_smoothing'] = usa_df.ewm(alpha=0.2)['Sales'].mean().shift(1)
# Calculate exponentially weighted averages with a smoothing constant 0.5 for the 'new_cases' column
forecast_df['0.5_exponential_smoothing'] = usa_df.ewm(alpha=0.5)['Sales'].mean().shift(1)
forecast_df.iloc[:10]


# In[147]:


print(forecast_accuracy(forecast_df, 'sales', '0.2_exponential_smoothing'))


# In[148]:


print(forecast_accuracy(forecast_df, 'sales', '0.5_exponential_smoothing'))


# In[149]:


plt.figure(figsize=(16,8))

sns.lineplot(data=forecast_df, x='timestamp', y='sales', color='skyblue', alpha=0.5)
sns.lineplot(data=forecast_df, x='timestamp', y='0.2_exponential_smoothing', color='red', alpha=0.5, label='Exponential Smoothing with a=0.2')
sns.lineplot(data=forecast_df, x='timestamp', y='0.5_exponential_smoothing', color='orange', alpha=0.5, label='Exponential Smoothing with a=0.5')


# In[150]:


import statsmodels.api as sm

trend_endog = usa_df[['Sales']]
trend_exog = pd.DataFrame()
# Count the number of days from the starting date of the time series
trend_exog['Month'] = (usa_df['timestamp'] - usa_df['timestamp'].values[0]).dt.days
trend_exog = sm.add_constant(trend_exog, prepend=True)
trend_endog.head(), trend_exog.head()


# In[151]:


linear_mod = sm.OLS(trend_endog['Sales'], trend_exog)
linear_res = linear_mod.fit()
print(linear_res.summary())


# In[152]:


# Obtain all parameters related to the prediction from quad_res by calling get_prediction().summary_frame()
linear_pred = linear_res.get_prediction().summary_frame()
# Rename all the parameters
linear_pred.columns = ['pred_sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
# Add observed 'happiness' and 'obesity' to quad_pred from the sample 
linear_pred[['sales', 'timestamp']] = usa_df[['Sales', 'timestamp']]
linear_pred.head()


# In[153]:


def regression_plot(plot_data):
    '''
    This function plots the forrcasting line over the observed data points
    
    Parameters
    ----------
    plot_data : DataFrame
        Container of observations, predictions, confidence intervals, and prediction intevals.
        
    Outputs
    -------
    Figure
        A regression line over observations with confidence interval and prediction interval.
    '''
    plt.figure(figsize=(16,8))
    # Use the seaborn.lineplot() method to plot the curvilinear regression line between 'obesity' and 'happiness' in plot_data
    sns.lineplot(data=plot_data, x='timestamp', y='pred_sales', color='red', label='Prediction')

    # Use the seaborn.scatterplot() method to plot the observced data points for 'obesity' and 'happiness' in plot_data
    sns.lineplot(data=plot_data, x='timestamp', y='sales', color='blue', alpha=0.5, label='Observation')

    # Use the matplotlib.pyplot.fill_between() method to fill between the lower and the upper bounds of confidence interval (ci_lower, ci_upper)
    plt.fill_between(x=plot_data['timestamp'], y1=plot_data['ci_lower'], y2=plot_data['ci_upper'], color='teal', alpha=0.2)

    # Use the matplotlib.pyplot.fill_between() method to fill between the lower and the upper bounds of prediction interval (pi_lower, pi_upper)
    plt.fill_between(x=plot_data['timestamp'], y1=plot_data['pi_lower'], y2=plot_data['pi_upper'], color='skyblue', alpha=0.2)

regression_plot(linear_pred)


# In[154]:


print(forecast_accuracy(linear_pred, 'sales', 'pred_sales'))


# In[155]:


# Predict the number of confirmed cases in day 803 (4th April 2022)
linear_res.get_prediction([1, 12]).summary_frame()


# ## 6.2 Forecasting via Non-linear Regression

# In[156]:


trend_exog['Month^2'] = trend_exog['Month'] ** 2
trend_exog.head()


# In[157]:


nonlinear_mod = sm.OLS(trend_endog['Sales'], trend_exog)
nonlinear_res = nonlinear_mod.fit()
print(nonlinear_res.summary())


# In[158]:


# Obtain all parameters related to the prediction from quad_res by calling get_prediction().summary_frame()
nonlinear_pred = nonlinear_res.get_prediction().summary_frame()
# Rename all the parameters
nonlinear_pred.columns = ['pred_sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
# Add observed 'happiness' and 'obesity' to quad_pred from the sample 
nonlinear_pred[['sales', 'timestamp']] = usa_df[['Sales', 'timestamp']]
nonlinear_pred.head()


# In[159]:


regression_plot(nonlinear_pred)


# In[160]:


print(forecast_accuracy(nonlinear_pred, 'sales', 'pred_sales'))


# In[161]:


# Concert 'new_tests' and 'new_vaccinations' into lagged variables by shifting for 1 day  
trend_exog[['sales_t-1']] = usa_df[['Sales']].shift(1)
trend_exog = trend_exog.dropna(axis=0)
trend_endog = trend_endog[1:]
trend_exog, trend_endog


# In[162]:


sales_mod = sm.OLS(trend_endog['Sales'], trend_exog)
sales_res = sales_mod.fit()
print(sales_res.summary())


# In[163]:


# Obtain all parameters related to the prediction from quad_res by calling get_prediction().summary_frame()
sales_pred = sales_res.get_prediction().summary_frame()
# Rename all the parameters
sales_pred.columns = ['pred_sales', 'pred_se', 'ci_lower', 'ci_upper', 'pi_lower', 'pi_upper']
sales_pred[['sales', 'timestamp']] = usa_df[['Sales', 'timestamp']]
sales_pred.head()


# In[164]:


nonlinear_res.get_prediction([1, 12,144]).summary_frame()

