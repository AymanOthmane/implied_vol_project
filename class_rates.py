import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import ustreasurycurve as ustcurve
import yfinance as yf
from scipy.interpolate import CubicSpline
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from data_toolbox import *


class rateCurve:
    """
    A class to manage and analyze yield curves based on a given as-of date.

    Attributes:
        str_as_of_date (str): The as-of date in string format (YYYY-MM-DD).
        as_of_date (datetime): The as-of date as a datetime object.
        yield_curve (DataFrame): The yield curve data for the as-of date.
        dt_yield_curve (DataFrame): The yield curve with datetime-based columns.
    """
    def __init__(self, as_of_date): 
        """
        Initialize the RateCurve object with the given as-of date.

        Args:
            as_of_date (str): The as-of date in 'YYYY-MM-DD' format.
        """    
        self.str_as_of_date = as_of_date
        self.as_of_date = dt.strptime(as_of_date,'%Y-%m-%d')
        self.yield_curve = self.get_yield_curve(as_of_date)
        self.dt_yield_curve = self.get_datetime_yield_curve(self.yield_curve.copy())

    def get_yield_curve(self, as_of_date):
        """
        Retrieve the yield curve data for the specified as-of date.

        Args:
            as_of_date (str): The as-of date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame: A DataFrame containing the yield curve data.
        """
        ustcurv = pd.DataFrame(ustcurve.nominalRates(as_of_date, as_of_date))
        ustcurv.index = ustcurv["date"]
        ustcurv = ustcurv.drop('date', axis=1)
        url =f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate={self.str_as_of_date}&endDate={self.str_as_of_date}&type=rate"

        # Make the API request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            sofr_data = response.json()  # Convert response to JSON
            
            # Convert to a Pandas DataFrame
            sofr_data = pd.DataFrame(sofr_data['refRates'])
            sofr = sofr_data.loc[:,'percentRate'].iloc[0]/100
        else:
            print("Error fetching SOFR data:", response.status_code)
            sofr = 0.043
        ustcurv.insert(0,'1d',sofr)
        return ustcurv
    
    def get_datetime_yield_curve(self, curve):
        """
        Convert the yield curve DataFrame to have datetime-based columns.

        Args:
            curve (DataFrame): The original yield curve DataFrame.

        Returns:
            DataFrame: A DataFrame with datetime objects as column names.
        """
        curve.rename(columns={
            "1d": self.as_of_date + relativedelta(days=1),
            "1m": self.as_of_date + relativedelta(months=1),
            "2m": self.as_of_date + relativedelta(months=2),
            "3m": self.as_of_date + relativedelta(months=3),
            "6m": self.as_of_date + relativedelta(months=6),
            "1y": self.as_of_date + relativedelta(years=1),
            "2y": self.as_of_date + relativedelta(years=2),
            "3y": self.as_of_date + relativedelta(years=3),
            "5y": self.as_of_date + relativedelta(years=5),
            "10y": self.as_of_date + relativedelta(years=10),
            "20y": self.as_of_date + relativedelta(years=20),
            "30y": self.as_of_date + relativedelta(years=30)
            }, inplace=True)
        return curve
        
    def get_rate(self, date):
        """
        Get the yield rate for a specific date using interpolation if necessary.

        Args:
            date (str): The date for which to retrieve the yield rate in 'YYYY-MM-DD' format.

        Returns:
            float: The interpolated yield rate for the specified date.
        """
        date = dt.strptime(date,'%Y-%m-%d') if isinstance(date, str) else date
        if date in self.dt_yield_curve.columns:
            return self.dt_yield_curve.loc[:,date]
        else:
            x = [date.toordinal() for date in self.dt_yield_curve.columns]
            y = self.dt_yield_curve.loc[self.str_as_of_date].tolist()
            cs = CubicSpline(x,y)
            date = [date.toordinal()]
            cs_rate = cs(date)
            return cs_rate

    def plot_curve(self):
        return self.yield_curve.transpose().plot()
    
    def get_option_implied_rate(self, S, C, P, K, T):
        """
        Calculate the risk-free rate using put-call parity.

        Args:
            S (float): Current price of the underlying asset.
            K (float): Strike price of the options.
            T (float): Time to maturity (in years).
            C (float): Price of the call option.
            P (float): Price of the put option.

        Returns:
            float: The implied risk-free rate.
        """
        rate = -(1 / T) * np.log((S - (C - P)) / K)
        return rate
