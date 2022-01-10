# pip install MetaTrader5
# pip install pandas-ta
# pip install TA-Lib
# pip install plotly
# pip install ipympl 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import pandas_ta as ta
import os as os
import plotly.graph_objects as go

%matplotlib inline
def get_mt5_time_frame(tf):
    if tf == 2:
        return mt5.TIMEFRAME_M2
    elif tf == 5:
        return mt5.TIMEFRAME_M5
    elif tf == 10:
        return mt5.TIMEFRAME_M10
    elif tf == 15:
        return mt5.TIMEFRAME_M15
    elif tf == 60:
        return mt5.TIMEFRAME_H1
    elif tf == 120:
        return mt5.TIMEFRAME_H2
    elif tf == "D":
        return mt5.TIMEFRAME_D1

def get_tf_name_alias(tf):
    if tf == 2:
        return "2M"
    elif tf == 5:
        return "5M"
    elif tf == 10:
        return "10M"
    elif tf == 15:
        return "15M"
    elif tf == 60:
        return "1H"
    elif tf == 120:
        return "2H"
    elif tf == "D":
        return "D"

# Baixa os dados dos ativos da bolsa e salva em CSV
def generate_database(tf,quantity_of_candles):
    mt5.initialize();
    with open('./bolsa.csv') as reader:
    
        lines = reader.readlines();
        # trocar qtde de intervalos que quiser
        
        # configurar timeframe desejado 
        mt5_tf = get_mt5_time_frame(tf)
        tf_alias = get_tf_name_alias(tf)
        for l in lines:
            ticker = l.translate({ord('\n'): None})
            # print(f"Importando {ticker}")
            try:
                importedData =  pd.DataFrame(mt5.copy_rates_from_pos(ticker, mt5_tf, 0, quantity_of_candles))
                if(not importedData.empty):
                    # Marca a coluna 'time' como indexador do DataFrame
                    importedData.set_index(pd.to_datetime(importedData['time'], unit='s'), inplace=True)
                    
                    # Ajusta o volume e calcula media + bb upper
                    importedData['volume'] = importedData['real_volume'].values / 1000
                    vol_bbands = ta.bbands(importedData['volume'],20,1)
                    importedData['volume_sma20'] = vol_bbands['BBM_20_1.0']
                    importedData['volume_bb_upper'] = round(vol_bbands['BBU_20_1.0'])
                    
                    # Calcula SMA20, Bollinger e adiciona ao dataframe
                    df_bbands = ta.bbands(importedData['close'],20,2)
                    importedData['sma20'] = df_bbands['BBM_20_2.0']
                    importedData['bb_lower'] = df_bbands['BBL_20_2.0']
                    importedData['bb_upper'] = df_bbands['BBU_20_2.0']
                    
                    # ADX
                    adx = ta.adx(importedData['high'], importedData['low'], importedData['close'])
                    importedData['adx_14'] = adx['ADX_14']
                    importedData['adx_di+'] = adx['DMP_14']
                    importedData['adx_di-'] = adx['DMN_14']

                    # Limpa colunas nao utilizadas
                    importedData = importedData.drop(columns=['spread', 'real_volume'])
                    # print(f"saveToFile step=start file=./data/{ticker}-{tf_alias}.db.csv") 
                    importedData.to_csv(f"./data/{ticker}-{tf_alias}.db.csv")
                    # print(f"saveToFile step=success file=./data/{ticker}-{tf_alias}.db.csv")
                else:
                    print(f"saveToFile step=empty file=./data/{ticker}-{tf_alias}.db.csv")
            except Exception as ex:
                print(ex)
                print(f"saveToFile step=error ticker={ticker}")
    print(f"saveToFile step=end")
    mt5.shutdown()

def calc_zscores(mean_size):
    with open('./bolsa.csv') as reader:
        # Params
        lines = reader.readlines();
        zscores = pd.DataFrame()
        # configurar timeframe desejado 
        tf = "D"
        mt5_tf = get_mt5_time_frame(tf)
        tf_alias = get_tf_name_alias(tf)
        
        for l in lines:
            try:
                ticker = l.translate({ord('\n'): None})
                pathToFile = f"./data/{ticker}-{tf_alias}.db.csv"
                #print(f"openFile step=start file={pathToFile}")
                # Nem todos os arquivos da lista tem dados (exemplo papeis de-listados que ainda estao no sistema ex:VVIT4)
                
                if(os.path.exists(pathToFile)):
                    fromCSV = pd.read_csv(pathToFile)
                    closePrice = pd.DataFrame();
                    if(not fromCSV.empty):
                        # Apenas a coluna de preco de fechamento
                        closePrice[ticker] = fromCSV['close']
                        calcs = pd.DataFrame();
                        calcs[f"{ticker}close"] = closePrice[ticker]
                        calcs[f"{ticker}mean"] = closePrice.loc[:, ticker].rolling(window=mean_size).mean()
                        calcs[f"{ticker}stddev"] = closePrice.loc[:, ticker].rolling(window=mean_size).std(ddof=0)
                        # ZScore = (Fechamento - media)/desvio
                        zscores[ticker] = (calcs[f"{ticker}close"] - calcs[f"{ticker}mean"]) / calcs[f"{ticker}stddev"] 
                        last = len(zscores[ticker]) -1
                        # Acima de dois desvios da media 
                        if(zscores[ticker][last] >= 2.0 or zscores[ticker][last] <= -2.0):
                            print(f"{ticker}\tmean={mean_size}\tz-score={zscores[ticker][last]}")
               
            except Exception as ex:
                print(ex) # Do nothing
                #print(f"openFile step=error file={pathToFile}")
                
def load_from_db(ticker, tf):
    print(f"m=load_from_db step=start {ticker}")
    tf_alias = get_tf_name_alias(tf)
    pathToFile = f"./data/{ticker}-{tf_alias}.db.csv"
    if(os.path.exists(pathToFile)):
        fromCSV = pd.read_csv(pathToFile)
        if(not fromCSV.empty):
            print(f"m=load_from_db step=success {pathToFile}")
            return fromCSV
        else:
            print(f"m=load_from_db step=empty {pathToFile}")
            return None;
        
def build_fig(someDf):
    fig = go.Figure(data=[go.Candlestick(x=someDf['time'],
                open=someDf['open'],
                high=someDf['high'],
                low=someDf['low'],
                close=someDf['close'])])
    fig.update_layout(height=1000)
    return fig


generate_database("D", 300)
df = load_from_db("B3SA3", "D")
df = df.drop(columns = ['time.1'])
df.tail(15)

with open('./bolsa.csv') as reader:
    # Params
    tf = "D"    
    tf_alias = get_tf_name_alias(tf)
    lines = reader.readlines();
    for l in lines:
        ticker = l.translate({ord('\n'): None})
        pathToFile = f"./data/{ticker}-{tf_alias}.db.csv"
        # print(pathToFile)
        if(os.path.exists(pathToFile)):
            fromCSV = pd.read_csv(pathToFile)
            lastIndex = len(fromCSV) -1 
            # Vazou a bollinger pra cima
            if(lastIndex > 0 and fromCSV['close'][lastIndex] >= fromCSV['bb_upper'][lastIndex]):
                if(fromCSV['adx_14'][lastIndex] > 35):
                    print(f"{fromCSV['time'][lastIndex]}\t{ticker}\tstatus=trend\tclose={fromCSV['close'][lastIndex]:.2f}\tbb_upper={fromCSV['bb_upper'][lastIndex]:.2f}\tadx={fromCSV['adx_14'][lastIndex]:.2f}\tdi+={fromCSV['adx_di+'][lastIndex]:.2f}\tdi-={fromCSV['adx_di-'][lastIndex]:.2f}".format(float))
                else:
                    print(f"{fromCSV['time'][lastIndex]}\t{ticker}\tstatus=pullback\tclose={fromCSV['close'][lastIndex]:.2f}\tbb_upper={fromCSV['bb_upper'][lastIndex]:.2f}\tadx={fromCSV['adx_14'][lastIndex]:.2f}\tdi+={fromCSV['adx_di+'][lastIndex]:.2f}\tdi-={fromCSV['adx_di-'][lastIndex]:.2f}".format(float))



#####
# Transforma para o modelo de dados que o Cerebro consome
ticker = "VALE3"
df = load_from_db(f"{ticker}", "D")
df = df.drop(columns = ['time.1','tick_volume'])
df.to_csv(f"{ticker}.csv", index = False)

bbands = btind.BBands(df['close'], period=20, devfactor=2)

df_to_plot = df[['time','close']]
df_to_plot.set_index('time', inplace=True)

%matplotlib widget
plt.grid(True)
plt.title(f"{ticker}")
plt.xticks(rotation=45)

# Plotando um periodo menor pq ficou pesado
plt.plot(df_to_plot.iloc[480:500], label="Preço")
plt.plot(df_to_plot.iloc[480:500], "ro")
plt.show()
######
# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        #self.log("m=notify_order")
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None
        
    def notify_trade(self, trade):
        self.log(f"m=notify_trade")
        help(trade)

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                    # current close less than previous close

                    if self.dataclose[-1] < self.dataclose[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        #self.log('BUY CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                #self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


cerebro = bt.Cerebro()
cerebro.broker.setcash(100000.0)

# Add a strategy
cerebro.addstrategy(LarryWilliams_9_1)
# Add broker commissions
# 0.1% ... divide by 100 to remove the %
cerebro.broker.setcommission(commission=0.001)

# Carrega o DF
data = bt.feeds.GenericCSVData(
    dataname=f"./{ticker}.csv",
    nullvalue=0.0,
    dtformat=('%Y-%m-%d'),
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1
)
# Pass it to the backtrader datafeed and add it to the cerebro
# Futuro tentar usar o DF carregado direto
# data = bt.feeds.PandasData(dataname=dataframe)

cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
cerebro.adddata(data)
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())


cerebro.run()
# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
%matplotlib widget
fig = cerebro.plot(iplot=False, figsize=(1280,720))[0][0]