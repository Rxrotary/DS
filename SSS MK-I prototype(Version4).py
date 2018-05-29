
# coding: utf-8

# ### MK-I Goal
#     1. Format the loop to only extract tickers with designated 'start' & 'end' dates.
#         -completed
#         
#     2. Adopt imputation technique to NaN and possibly add pipeline.
#         -this imputation garbage didn't make any difference
#         
#     3. Add Kalman Filter as Feature.
#         -completed
#         
#     4. Feature Importance to pick important features.
#         -completed for randomforestclassifier
#     
#     5. Grid_search for more consistent results.
#         -completed
#         
#     6. Adding more technical indicators.
#         -completed

# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import datetime as dt
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import requests
import pickle
import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
from sklearn import model_selection
import multiprocessing

# datetime : to specify dates for the Pandas datareader 
# os : to check for, and create, directories


# In[3]:


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers)
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

save_sp500_tickers()


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pykalman import KalmanFilter


# In[5]:


alphavant = 'UI33NK6C8DC1H4WM'


# In[6]:


def get_data(ticker):

    # Technical Indicators
    ti = TechIndicators(key=alphavant, output_format='pandas')
    sma, _ = ti.get_sma(symbol=ticker, interval='daily', time_period=15, series_type='close') 
    ema, _ = ti.get_ema(symbol=ticker, interval='daily', time_period=12, series_type='close')
    macd, _ = ti.get_macd(symbol=ticker, interval='daily', series_type='close')
    stoch, _ = ti.get_stoch(symbol=ticker, interval='daily')
    rsi, _ = ti.get_rsi(symbol=ticker, interval='daily', time_period=14, series_type='close')
    adx, _ = ti.get_adx(symbol=ticker, interval='daily', time_period=14)
    cci, _ = ti.get_cci(symbol=ticker, interval='daily', time_period=20)
    aroon, _ = ti.get_aroon(symbol=ticker, interval='daily', time_period=25, series_type='close')
    bbands, _ = ti.get_bbands(symbol=ticker, interval='daily', time_period=20, series_type='close')
       
    # new
    ad, _ = ti.get_ad(symbol=ticker, interval='daily')
    obv, _ = ti.get_obv(symbol=ticker, interval='daily')
    ppo, _ = ti.get_ppo(symbol=ticker, interval='daily', series_type='close')
    mom, _ = ti.get_mom(symbol=ticker, interval='daily', time_period=10, series_type='close')
    roc, _ = ti.get_roc(symbol=ticker, interval='daily', time_period=10, series_type='close')
    ultosc, _ = ti.get_ultosc(symbol=ticker, interval='daily')
    dx, _ = ti.get_dx(symbol=ticker, interval='daily', time_period=14)
    sar, _ = ti.get_sar(symbol=ticker, interval='daily')
    atr, _ = ti.get_atr(symbol=ticker, interval='daily', time_period=10)
    willr, _ = ti.get_willr(symbol=ticker, interval='daily', time_period=14)
    
    
    tech_ind = pd.concat([sma, ema, macd, stoch, rsi, adx, cci, aroon, bbands, ad, obv, ppo, mom, roc, ultosc, dx, sar, atr, willr], axis=1)

    
    ts = TimeSeries(key=alphavant, output_format='pandas')
    close = ts.get_daily(symbol=ticker, outputsize='full')[0]['4. close']   # compact/full
    direction = (close > close.shift()).astype(int)      # comparing to the closing price of the previous day
    target = direction.shift(-1).fillna(0).astype(int)   # target:whether the closing price went up the next day
    target.name = 'target'
    # 0 if the price went down the next day
    # 1 if the price went up the next day

    data = pd.concat([tech_ind, close, target], axis=1)

    return data


# In[7]:


# exponential moving average:Past values have a diminishing contribution to the average, 
        #while more recent values have a greater contribution. This method allows the moving average 
        #to be more responsive to changes in the data

# moving average convergence / divergence: calculates the difference between 
        #an instrument's 26-day and 12-day exponential moving averages (EMA)

# stochastic oscillator:The stochastic oscillator is plotted within a range of zero and 100 
        #and signals overbought conditions above 80 and oversold conditions below 20.
        #by default, fastkperiod=5, slowkperiod=3

# relative strength index:RSI measures the velocity and magnitude of directional price moves 
        #and represents the data graphically by oscillating between 0 and 100. The indicator is 
        #calculated using the average gains and losses of an asset over a specified time period.

# average directional movement index: used to quantify trend strength. ADX calculations are 
        #based on a moving average of price range expansion over a given period of time.

# commodity channel index:it's trying to predict a cycle using moving averages, the more attuned 
        #the moving average amounts (days averaged) are to the cycle, the more accurate the average will be.

# Aroon: The indicator's greatest value is in helping traders and investors to distinguish whether a 
        #long-term trend is ending or simply stalling before another move.

# Bollinger bands: employ upper and lower standard deviation bands together with a center 
        #simple moving average band around price to identify a stock's high and low volatility points
        #When the price is near the upper or lower band it indicates that a reversal may be imminent. 
        #The middle band becomes a support or resistance level. The upper and lower bands can also be 
        #interpreted as price targets. When the price bounces off of the lower band and crosses the middle band, 
        #then the upper band becomes the price target.
        
# Advance/Decline Line: It is a cumulative total of the Advancing-Declining Issues indicator. The Advance/Decline Line of 
        # a market (such as the NYSE or NASDAQ) moves with the price of the market index. Look for agreement/divergence to 
        # confirm/deny price trends.
        
# On Balance Volume (OBV) is a cumulative total of the up and down volume. When the close is higher than the previous close,
        # the volume is added to the running total, and when the close is lower than the previous close, the volume is 
        # subtracted from the running total.

# Price Oscillator Percent shows the percentage difference between two moving averages. A buy signal is generate when the 
        # Price Oscillator Percent rises above zero, and a sell signal when the it falls below zero.

# Momentum is a measurement of the acceleration and deceleration of prices. It indicates if prices are increasing at 
        # an increasing rate or decreasing at a decreasing rate. The Momentum function can be applied to the price, or to 
        # any other data series. 

# Rate of Change function measures rate of change relative to previous periods. The function is used to determine how rapidly 
        # the data is changing.
    
# Ultimate Oscillator is the weighted sum of three oscillators of different time periods. The typical time periods are 7, 14 
        # and 28. The values of the Ultimate Oscillator range from zero to 100. Values over 70 indicate overbought conditions, 
        # and values under 30 indicate oversold conditions.
        
# DX is usually smoothed with a moving average (i.e. the ADX). The values range from 0 to 100, but rarely get above 60. 
        # To interpret the DX, consider a high number to be a strong trend, and a low number, a weak trend.
    
# Parabolic SAR calculates a trailing stop. Simply exit when the price crosses the SAR. The SAR assumes that you are 
        # always in the market, and calculates the Stop And Reverse point when you would close a long position and open 
        # a short position or vice versa.
        
# ATR is a Welles Wilder style moving average of the True Range. The ATR is a measure of volatility. High ATR values 
        # indicate high volatility, and low values indicate low volatility, often seen when the price is flat. 
    
# Williams %R is similar to an unsmoothed Stochastic %K. The values range from zero to 100, and are charted on an inverted scale,
        # that is, with zero at the top and 100 at the bottom. Values below 20 indicate an overbought condition and a sell signal
        # is generated when it crosses the 20 line. Values over 80 indicate an oversold condition and a buy signal is generated 
        # when it crosses the 80 line.
        
### http://www.fmlabs.com/reference/default.htm?url=StochasticOscillator.htm
    


# In[8]:


def get_data_from_AlphaV(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_list'):
        os.makedirs('stock_list')
    
    for ticker in tickers[:]:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_list/{}.csv'.format(ticker)):
            df = get_data(ticker)
            df.to_csv('stock_list/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


# In[9]:


get_data_from_AlphaV()


# In[10]:


def rebalance(unbalanced_data):

    # Separate majority and minority classes
    data_minority = unbalanced_data[unbalanced_data.target==0]
    data_majority = unbalanced_data[unbalanced_data.target==1]

    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)

    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    data_upsampled.sort_index(inplace=True)

    # Display new class counts
    data_upsampled.target.value_counts()

    return data_upsampled


# In[11]:


def normalize(x):
    # Standardize features by removing the mean and scaling to unit variance.
    #Centering and scaling happen independently on each feature by computing the relevant statistics 
    #on the samples in the training set. Mean and standard deviation are then stored to be used on 
    #later data using the transform method. Standardization of a dataset is a common requirement for 
    #many machine learning estimators: they might behave badly if the individual feature do not more 
    #or less look like standard normally distributed data. 
    
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)

    return x_norm


# In[12]:


def scores(models, X, y):
    l=[]
    for model in models:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred) 
            # The F1 score is the harmonic average of the precision and recall, 
            #where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
        auc = roc_auc_score(y, y_pred)
            #accuracy = {ticker: {model: {'Accuracy Score: {0:0.2f} %'.format(acc * 100)}}}
            #f1score = {ticker: {model: {"F1 Score: {0:0.4f}".format(f1)}}}
            #roccurve = {ticker: {model: {"Area Under ROC Curve Score: {0:0.4f}".format(auc) }}}
        l.append([ticker, model.__class__.__name__, acc, f1, auc]) 

    return pd.DataFrame.from_dict(l)


# In[13]:


models = [GaussianNB(),
          SVC(random_state=5),
          RandomForestClassifier(random_state=5),
          MLPClassifier(random_state=5)]
models


# In[15]:


tickers = save_sp500_tickers()

for ticker in tickers:               
    if not os.path.exists('KFresults'):
        os.makedirs('KFresults')
    
    try:
        os.path.exists('stock_list/{}.csv'.format(ticker))
        df = pd.read_csv('stock_list/{}.csv'.format(ticker))
        df.set_index('Unnamed: 0', inplace=True)
    except OSError:
        print('File {} Does Not Exist'.format(ticker))
        pass

    df = df.rename(index=str, columns={'4. close':'Close'})
    #### applying Kalman Filter

    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(df['Close'])
    state_means = pd.Series(state_means.flatten(), index=df.index)
    df = pd.concat([df, state_means], axis=1)
    df = df.rename(index=str, columns={0:'KalmanFilter'})

    start ='2015-01-01'
    end ='2017-12-31'

    #df.loc[start:end]

    data = df.fillna(df.mean())
    data_train = data[start:end]
    data_train = rebalance(data_train)
    y = data_train.target
    X = data_train.drop('target', axis=1)
    X = normalize(X)

    data_val = data[end:]
    y_val = data_val.target
    X_val = data_val.drop('target', axis=1)
    X_val = normalize(X_val)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6)

    for model in models:
        model.fit(X_train, y_train)

    ds = scores(models, X_test, y_test)

    # just in case your connection breaks, we'd like to save our progress!
    if not os.path.exists('KFresults/{}.csv'.format(ticker)):
        ds.to_csv('KFresults/{}.csv'.format(ticker))
    else:
        print('Already have {}'.format(ticker))


# In[ ]:


def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers):
        
        try:
            os.path.exists('KFresults/{}.csv'.format(ticker))
            df = pd.read_csv('KFresults/{}.csv'.format(ticker))
        except OSError:
            pass

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.append(df)

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    
    main_df.rename(columns={'0': 'Ticker', '1':'Model', '2':'Accuracy', '3':'F1', '4':'ROC'}, inplace=True)
    
    main_df.to_csv('sp500_join_KFresults.csv')


# In[39]:


compile_data()


# In[ ]:


################################################################################################################################


# In[111]:


summary = pd.read_csv('sp500_join_KFresults.csv')
summary = summary.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)
summary.nlargest(10, 'Accuracy')


# In[112]:


summary.max()


# In[115]:


summary.nlargest(10,'Accuracy')


# In[10]:


# select models to run gridsearch, 
models = [SVC(),
          RandomForestClassifier()]


# In[22]:


tickers = save_sp500_tickers()

for ticker in tickers:               
    if not os.path.exists('grid_results'):
        os.makedirs('grid_results')
    
    try:
        os.path.exists('stock_list/{}.csv'.format(ticker))
        df = pd.read_csv('stock_list/{}.csv'.format(ticker))
        df.set_index('Unnamed: 0', inplace=True)
    except OSError:
        print('File {} Does Not Exist'.format(ticker))
        pass
    

    df = df.rename(index=str, columns={'4. close':'Close'})
    #### applying Kalman Filter

    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(df['Close'])
    state_means = pd.Series(state_means.flatten(), index=df.index)
    df = pd.concat([df, state_means], axis=1)
    df = df.rename(index=str, columns={0:'KalmanFilter'})

    start ='2015-01-01'
    end ='2017-12-31'

    #df.loc[start:end]

    data = df.fillna(df.mean())
    data_train = data[start:end]
    data_train = rebalance(data_train)
    y = data_train.target
    X = data_train.drop('target', axis=1)
    X = normalize(X)

    data_val = data[end:]
    y_val = data_val.target
    X_val = data_val.drop('target', axis=1)
    X_val = normalize(X_val)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6)

    for model in models:
        model.fit(X_train, y_train)

    ds = scores(models, X_test, y_test)

        # just in case your connection breaks, we'd like to save our progress!
    if not os.path.exists('grid_results/{}.csv'.format(ticker)):
        dgrid.to_csv('grid_results/{}.csv'.format(ticker))
    else:
        print('Already have {}'.format(ticker))
        continue
    
    # Grid search
    grid_data = [[{'kernel': ['poly'], 'degree': [1, 2, 3, 4], 'C': [0.1, 1, 10, 100], 'random_state': [5]},
                  {'kernel': ['rbf', 'sigmoid'], 'C': [0.1, 1, 10, 100], 'random_state': [5]}],
                  {'n_estimators': [10, 50, 100],
                   'criterion': ['gini', 'entropy'],
                   'max_depth': [None, 10, 50, 100],
                   'min_samples_split': [2, 5, 10],
                   'random_state': [5]},
                  {'hidden_layer_sizes': [10, 50, 100],
                   'activation': ['identity', 'logistic', 'tanh', 'relu'],
                   'solver': ['lbfgs', 'sgd', 'adam'],
                   'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'max_iter': [200, 400, 800],
                   'random_state': [5]}]
    models_grid = list()

    for i in range(len(models)):
        grid = GridSearchCV(models[i], grid_data[i], scoring='f1', n_jobs=4).fit(X_train, y_train)
        print(grid.best_params_)
        model = grid.best_estimator_
        models_grid.append(model)

    # Validation data
    dgrid = scores(models_grid, X_val, y_val)

    dgrid.to_csv('grid_results/{}'.format(ticker))
              
  


# In[24]:


def compile_data_grid():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers):
        
        try:
            os.path.exists('grid_results/{}.csv'.format(ticker))
            df = pd.read_csv('grid_results/{}.csv'.format(ticker))
        except OSError:
            pass

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.append(df)

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    
    main_df.rename(columns={'0': 'Ticker', '1':'Model', '2':'Accuracy', '3':'F1', '4':'ROC'}, inplace=True)
    
    main_df.to_csv('sp500_grid_results.csv')


# In[25]:


compile_data_grid()


# In[16]:


# feature importance for random forest classifier

rf = RandomForestClassifier(random_state=5)

## Fit the model on your training data.
rf.fit(X_train, y_train) 
## And score it on your testing data.
rf.score(X_test, y_test)

feature_importances = pd.DataFrame(rf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)


# In[17]:


feature_importances.importance


# In[65]:


rf.feature_importances_


# In[19]:


x = (feature_importances.index)
labels = X.columns


# In[26]:


plt.bar(x, feature_importances.importance*100, color='r')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Importance %', fontsize=15)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()


# In[29]:


grid_summary = pd.read_csv('sp500_grid_results.csv')


# In[30]:


grid_summary.head()


# In[31]:


grid_summary = grid_summary.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)


# In[113]:





# In[32]:


grid_summary.describe()


# In[34]:


grid_summary.loc[grid_summary['Accuracy'].idxmax()]   #Top1


# In[35]:


grid_summary.loc[grid_summary['F1'].idxmax()]


# In[36]:


grid_summary.loc[grid_summary['ROC'].idxmax()]   


# In[109]:


grid_summary.max()


# In[107]:


grid_summary.quantile(.90)


# In[104]:


grid_summary.nlargest(10, 'Accuracy')  #Top10


# In[105]:


grid_summary.nlargest(20, 'F1')


# In[41]:


grid_summary.nlargest(10, 'ROC')

