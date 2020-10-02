#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff


# In[2]:


# Importing dataset and examining its descriptive statistics
dataset = pd.read_csv("EireStay.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# In[3]:


# Converting Categorical features into Numerical features
dataset['meal'] = dataset['meal'].map({'BB':0, 'FB':1, 'HB':2, 'SC':3, 'Undefined':4})
dataset['market_segment'] = dataset['market_segment'].map({'Direct':0, 'Online TA':1})
dataset['reserved_room_type'] = dataset['reserved_room_type'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8})
dataset['deposit_type'] = dataset['deposit_type'].map({'No Deposit':0, 'Non Refund':1, 'Refundable':2})


# In[4]:


# Plotting Correlation Heatmap
#No redundant variables were found
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')


# In[5]:


#Convert variables like lead_time, booking_changes, days_in_waiting_list and average daily rate into categorical
#On the basis of low and high - low being below average or equal to average, high being above average - categories will be based on median too if need be
#Lead_Time

def converter1(column):
    if column <=25:
        return "Low"
    else:
        return "High"

dataset['lead_time_category'] = dataset['lead_time'].apply(converter1)

def converter2(column):
    if column <=0.315725:
        return "Low"
    else:
        return "High"

dataset['booking_changes_category'] = dataset['booking_changes'].apply(converter2)

def converter3(column):
    if column <=0.025653:
        return "Low"
    else:
        return "High"

dataset['waiting_list_category'] = dataset['days_in_waiting_list'].apply(converter3)


# In[6]:


dataset.head()


# In[7]:


#Map the lead time, booking changes and wait list category into 0's and 1's - 0's being low and 1 being high
dataset['lead_time_category'] = dataset['lead_time_category'].map({'Low':0, 'High':1})
dataset['waiting_list_category'] = dataset['waiting_list_category'].map({'Low':0, 'High':1})
dataset['booking_changes_category'] = dataset['booking_changes_category'].map({'Low':0, 'High':1})


# In[8]:


#Create a new variable for guest category -> 2 labels - adults only, adults, children and babies


# In[9]:


dataset['Kids'] = dataset['babies'] + dataset['children']


# In[10]:


def converter1(column):
    if column >0:
        return "Adults and Children"
    else:
        return "Adults only"

dataset['Guest_Category'] = dataset['Kids'].apply(converter1)


# In[11]:


dataset['Guest_Category'] = dataset['Guest_Category'].map({'Adults only':0, 'Adults and Children':1})


# In[12]:


def converter1(column):
    if column<=2:
        return "Low"
    else:
        return "High"

dataset['stays_in_week_nights_category'] = dataset['stays_in_week_nights'].apply(converter1)

def converter2(column):
    if column<=0:
        return "Low"
    else:
        return "High"

dataset['stays_in_weekend_nights_category'] = dataset['stays_in_weekend_nights'].apply(converter2)


# In[13]:


dataset['stays_in_weekend_nights_category'] = dataset['stays_in_weekend_nights_category'].map({'Low':0, 'High':1})


# In[14]:


dataset['stays_in_week_nights_category'] = dataset['stays_in_week_nights_category'].map({'Low':0, 'High':1})


# In[15]:


def converter3(column):
    if column <=0:
        return "Low"
    else:
        return "High"

dataset['previous_stays_category'] = dataset['previous_stays'].apply(converter3)


# In[16]:


dataset['previous_stays_category'] = dataset['previous_stays_category'].map({'Low':0, 'High':1})


# In[17]:


def converter3(column):
    if column <=0.920119:
        return "Low"
    else:
        return "High"

dataset['total_of_special_requests_category'] = dataset['total_of_special_requests'].apply(converter3)


# In[18]:


dataset['total_of_special_requests_category'] = dataset['total_of_special_requests_category'].map({'Low':0, 'High':1})


# In[19]:


def converter4(column):
    if column <=87.680000:
        return "Low"
    else:
        return "High"

dataset['average_daily_rate_category'] = dataset['average_daily_rate'].apply(converter4)


# In[20]:


dataset['average_daily_rate_category'] = dataset['average_daily_rate_category'].map({'Low':0, 'High':1})


# In[21]:


X = dataset


# In[22]:


# Dividing data into subsets
#Specific booking details related to aspects like the method of booking, deposit type, previous stays, waiting list, etc.
subset1 = X[['market_segment','deposit_type','booking_changes', 'days_in_waiting_list', 'lead_time', 'average_daily_rate']]

#Booking details specific to the resort and its facilities and customer information
subset2 = X[['stays_in_weekend_nights','stays_in_week_nights','adults', 'children', 'babies', 'meal', 'reserved_room_type', 'total_of_special_requests', 'previous_stays']]


# In[23]:


#This subset comprises of the Guest Category variable with all the other variables in subset 2
subset2_1 = X[['stays_in_weekend_nights','stays_in_week_nights','Guest_Category', 'meal', 'reserved_room_type', 'total_of_special_requests', 'previous_stays']]


# In[24]:


#This subset comprises of the Stay in weekend nights category variable with all the other variables in subset 2
subset2_2 = X[['stays_in_weekend_nights_category','stays_in_week_nights','adults', 'children', 'babies', 'meal', 'reserved_room_type', 'total_of_special_requests', 'previous_stays']]


# In[25]:


#This subset comprises of the Stay in week nights category variable with all the other variables in subset 2
subset2_3 = X[['stays_in_weekend_nights','stays_in_week_nights_category','adults', 'children', 'babies', 'meal', 'reserved_room_type', 'total_of_special_requests', 'previous_stays']]


# In[26]:


#This subset comprises of the Previous stays category variable with all the other variables in subset 2
subset2_4 = X[['stays_in_weekend_nights','stays_in_week_nights','adults', 'children', 'babies', 'meal', 'reserved_room_type', 'total_of_special_requests', 'previous_stays_category']]


# In[27]:


#This subset comprises of the Total of Special Requests category variable with all the other variables in subset 2
subset2_5 = X[['stays_in_weekend_nights','stays_in_week_nights','adults', 'children', 'babies', 'meal', 'reserved_room_type', 'total_of_special_requests_category', 'previous_stays']]


# In[28]:


# Normalizing numerical features so that each feature has mean 0 and variance 1 for subsets 1, 2, 2.1, 2.2, 2.3, 2.4 and 2.5
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X21 = feature_scaler.fit_transform(subset2_1)
X22 = feature_scaler.fit_transform(subset2_2)
X23 = feature_scaler.fit_transform(subset2_3)
X24 = feature_scaler.fit_transform(subset2_4)
X25 = feature_scaler.fit_transform(subset2_5)


# In[33]:


# Analysis on subset2
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[37]:


#Start with 5 clusters for meal type as this categorical variable has 5 labels (levels) - to see if distinct clusters are formed 
#on the basis of the labels of this variable

kmeans = KMeans(n_clusters = 5)
kmeans.fit(X2)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=50,n_iter=2000)
x_tsne = tsne.fit_transform(X2)

weekend = list(X['stays_in_weekend_nights'])
weekdays = list(X['stays_in_week_nights'])
adults = list(X['adults'])
children = list(X['children'])
babies = list(X['babies'])
meal = list(X['meal'])
room = list(X['reserved_room_type'])
specialrequests = list(X['total_of_special_requests'])
previousstays = list(X['previous_stays'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'weekend: {a}; weekdays: {b}; adults:{c}; children:{d}; babies:{e}; meal:{f}; room:{g}; specialrequests:{h}; previous_stays:{i}' for a,b,c,d,e,f,g,h,i in list(zip(weekend,weekdays,adults,children,babies,meal,room,specialrequests,previousstays))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-MealType.html')


# In[38]:


#Start with 9 clusters for reserved room type as this categorical variable has 9 labels (levels) - to see if distinct clusters are formed 
#on the basis of the labels of this variable
kmeans = KMeans(n_clusters = 9)
kmeans.fit(X2)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=20,n_iter=2000)
x_tsne = tsne.fit_transform(X2)

weekend = list(X['stays_in_weekend_nights'])
weekdays = list(X['stays_in_week_nights'])
adults = list(X['adults'])
children = list(X['children'])
babies = list(X['babies'])
meal = list(X['meal'])
room = list(X['reserved_room_type'])
specialrequests = list(X['total_of_special_requests'])
previousstays = list(X['previous_stays'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'weekend: {a}; weekdays: {b}; adults:{c}; children:{d}; babies:{e}; meal:{f}; room:{g}; specialrequests:{h}; previous_stays:{i}' for a,b,c,d,e,f,g,h,i in list(zip(weekend,weekdays,adults,children,babies,meal,room,specialrequests,previousstays))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-RoomType.html')


# In[52]:


#Start with 2 clusters for guest_category as this is a categorical variable, which is a combination of three numeric variables adults, babies and children) 
#has 2 labels (levels) - to see if distinct clusters are formed on the basis of the labels of ths variable
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X21)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=50,n_iter=2000)
x_tsne = tsne.fit_transform(X21)

weekend = list(X['stays_in_weekend_nights'])
weekdays = list(X['stays_in_week_nights'])
guest = list(X['Guest_Category'])
meal = list(X['meal'])
room = list(X['reserved_room_type'])
specialrequests = list(X['total_of_special_requests'])
previousstays = list(X['previous_stays'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'weekend: {a}; weekdays: {b}; guest:{c}; meal:{d}; room:{e}; specialrequests:{f}; previousstays:{g};' for a,b,c,d,e,f,g in list(zip(weekend,weekdays,guest,meal,room,specialrequests,previousstays))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-GuestType.html')


# In[29]:


#Start with 2 clusters for weekend nights category - as this is a categorical variable, which is based on the median number of weekend night stays 
#has 2 labels (levels) - see if distinct clusters are formed on the basis of the labels of ths variable
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X22)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=50,n_iter=2000)
x_tsne = tsne.fit_transform(X22)

weekend = list(X['stays_in_weekend_nights_category'])
weekdays = list(X['stays_in_week_nights'])
adults = list(X['adults'])
children = list(X['children'])
babies = list(X['babies'])
meal = list(X['meal'])
room = list(X['reserved_room_type'])
specialrequests = list(X['total_of_special_requests'])
previousstays = list(X['previous_stays'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'weekend: {a}; weekdays: {b}; adults:{c}; children:{d}; babies:{e}; meal:{f}; room:{g}; specialrequests:{h}; previousstays:{i}' for a,b,c,d,e,f,g,h,i in list(zip(weekend,weekdays,adults,children,babies,meal,room,specialrequests,previousstays))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-WeekendNightsType.html')


# In[63]:


#Start with 2 clusters for weekday nights category - as this is a categorical variable, which is based on the median number of weekday night stays
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X23)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=50,n_iter=2000)
x_tsne = tsne.fit_transform(X23)

weekend = list(X['stays_in_weekend_nights'])
weekdays = list(X['stays_in_week_nights_category'])
adults = list(X['adults'])
children = list(X['children'])
babies = list(X['babies'])
meal = list(X['meal'])
room = list(X['reserved_room_type'])
specialrequests = list(X['total_of_special_requests'])
previousstays = list(X['previous_stays'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'weekend: {a}; weekdays: {b}; adults:{c}; children:{d}; babies:{e}; meal:{f}; room:{g}; specialrequests:{h}; previousstays:{i}' for a,b,c,d,e,f,g,h,i in list(zip(weekend,weekdays,adults,children,babies,meal,room,specialrequests,previousstays))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-WeekNightsType.html')


# In[68]:


#Start with 2 clusters for previous_stay category - as this is a categorical variable, which is based on the median number of previous stays
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X24)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=50,n_iter=2000)
x_tsne = tsne.fit_transform(X24)

weekend = list(X['stays_in_weekend_nights'])
weekdays = list(X['stays_in_week_nights'])
adults = list(X['adults'])
children = list(X['children'])
babies = list(X['babies'])
meal = list(X['meal'])
room = list(X['reserved_room_type'])
specialrequests = list(X['total_of_special_requests'])
previousstays = list(X['previous_stays_category'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'weekend: {a}; weekdays: {b}; adults:{c}; children:{d}; babies:{e}; meal:{f}; room:{g}; specialrequests:{h}; previousstays:{i}' for a,b,c,d,e,f,g,h,i in list(zip(weekend,weekdays,adults,children,babies,meal,room,specialrequests,previousstays))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-PreviousStays.html')


# In[69]:


#Start with 2 clusters for special_requests category - as this is a categorical variable, which is based on the mean number of special requests
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X25)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=50,n_iter=2000)
x_tsne = tsne.fit_transform(X25)

weekend = list(X['stays_in_weekend_nights'])
weekdays = list(X['stays_in_week_nights'])
adults = list(X['adults'])
children = list(X['children'])
babies = list(X['babies'])
meal = list(X['meal'])
room = list(X['reserved_room_type'])
specialrequests = list(X['total_of_special_requests_category'])
previousstays = list(X['previous_stays'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'weekend: {a}; weekdays: {b}; adults:{c}; children:{d}; babies:{e}; meal:{f}; room:{g}; specialrequests:{h}; previousstays:{i}' for a,b,c,d,e,f,g,h,i in list(zip(weekend,weekdays,adults,children,babies,meal,room,specialrequests,previousstays))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SpecialRequests.html')


# In[29]:


# Analysis on subset1 
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[38]:


#The number of clusters can either be between 2 and 4
#Start with 2 and 3
#Try and see if clusters can be created on the basis of market segment or deposit type 
#Market segment has two levels - deposit type has three levels
#Perplexity of 50 looks best
# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=50,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

market = list(X['market_segment'])
deposit = list(X['deposit_type'])
bookchange = list(X['booking_changes'])
waitlist = list(X['days_in_waiting_list'])
leadtime = list(X['lead_time'])
avgdailyrate = list(X['average_daily_rate'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; deposit_type: {b}; booking_changes:{c}; days_in_waiting_list:{d}; lead_time:{e}; average_daily_rate:{f};' for a,b,c,d,e,f in list(zip(market,deposit,bookchange,waitlist,leadtime,avgdailyrate))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNEFinal.html')


# In[30]:


#Clusters were formed on the basis of market_segment
#Determining the optimal perplexity - start with 20
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=20,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

market = list(X['market_segment'])
deposit = list(X['deposit_type'])
bookchange = list(X['booking_changes'])
waitlist = list(X['days_in_waiting_list'])
leadtime = list(X['lead_time'])
avgdailyrate = list(X['average_daily_rate'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; deposit_type: {b}; booking_changes:{c}; days_in_waiting_list:{d}; lead_time:{e}; average_daily_rate:{f}; ' for a,b,c,d,e,f in list(zip(market,deposit,bookchange,waitlist,leadtime,avgdailyrate))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-20.html')


# In[275]:


#Clusters were formed on the basis of market_segment
#Determining the optimal perplexity - 25
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=25,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

market = list(X['market_segment'])
deposit = list(X['deposit_type'])
bookchange = list(X['booking_changes'])
waitlist = list(X['days_in_waiting_list'])
leadtime = list(X['lead_time'])
avgdailyrate = list(X['average_daily_rate'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; deposit_type: {b}; booking_changes:{c}; days_in_waiting_list:{d}; lead_time:{e}; average_daily_rate:{f}; ' for a,b,c,d,e,f in list(zip(market,deposit,bookchange,waitlist,leadtime,avgdailyrate))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-25.html')


# In[276]:


#Clusters were formed on the basis of market_segment
#Determining the optimal perplexity - 30 
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=30,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

market = list(X['market_segment'])
deposit = list(X['deposit_type'])
bookchange = list(X['booking_changes'])
waitlist = list(X['days_in_waiting_list'])
leadtime = list(X['lead_time'])
avgdailyrate = list(X['average_daily_rate'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; deposit_type: {b}; booking_changes:{c}; days_in_waiting_list:{d}; lead_time:{e}; average_daily_rate:{f}; ' for a,b,c,d,e,f in list(zip(market,deposit,bookchange,waitlist,leadtime,avgdailyrate))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-30.html')


# In[277]:


#Clusters were formed on the basis of market_segment
#Determining the optimal perplexity - 35
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=35,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

market = list(X['market_segment'])
deposit = list(X['deposit_type'])
bookchange = list(X['booking_changes'])
waitlist = list(X['days_in_waiting_list'])
leadtime = list(X['lead_time'])
avgdailyrate = list(X['average_daily_rate'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; deposit_type: {b}; booking_changes:{c}; days_in_waiting_list:{d}; lead_time:{e}; average_daily_rate:{f}; ' for a,b,c,d,e,f in list(zip(market,deposit,bookchange,waitlist,leadtime,avgdailyrate))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-35.html')


# In[278]:


#Clusters were formed on the basis of market_segment
#Determining the optimal perplexity - 40
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=40,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

market = list(X['market_segment'])
deposit = list(X['deposit_type'])
bookchange = list(X['booking_changes'])
waitlist = list(X['days_in_waiting_list'])
leadtime = list(X['lead_time'])
avgdailyrate = list(X['average_daily_rate'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; deposit_type: {b}; booking_changes:{c}; days_in_waiting_list:{d}; lead_time:{e}; average_daily_rate:{f}; ' for a,b,c,d,e,f in list(zip(market,deposit,bookchange,waitlist,leadtime,avgdailyrate))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-40.html')


# In[279]:


#Clusters were formed on the basis of market_segment
#Determining the optimal perplexity - 45
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity=45,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

market = list(X['market_segment'])
bookchange = list(X['booking_changes'])
waitlist = list(X['days_in_waiting_list'])
leadtime = list(X['lead_time'])
avgdailyrate = list(X['average_daily_rate'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; deposit_type: {b}; booking_changes:{c}; days_in_waiting_list:{d}; lead_time:{e}; average_daily_rate:{f}; ' for a,b,c,d,e,f in list(zip(market,deposit,bookchange,waitlist,leadtime,avgdailyrate))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-45.html')

