# Leitor de dados da plataforma MT5
# Le os dados e salva em arquivos csv para nao ter que ficar buscando os dados
# e dependendo sempre da plataforma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import pandas_ta as ta
import os as os
import plotly.graph_objects as go

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
    return fig;



# Tese
# Trade de reversao
# Fechou fora da Banda de Bollinger 2 sem ADX forte (< 30), faz operacao esperando pullback
# Stop = tamanho do candle gatilho plotado da extremidade