import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, bisect, newton
from datetime import datetime as dt
from data_toolbox import *
from class_rates import *

class SPX_Options:
    """
    A class designed to import, clean, and analyze SPX and SPX options data for a specified as-of date.

    This class provides functionality to process options data, calculate key metrics, and perform analysis
    based on a given date. It initializes with a specified as-of date and file path, retrieving and
    preparing the necessary data for further analysis.

    Attributes:
        str_as_of_date (str): The as-of date in string format (YYYY-MM-DD).
        as_of_date (datetime): The as-of date as a datetime object.
        path (str): The file path to the options data CSV file.
        options_data (DataFrame): The raw options data retrieved from the CSV file.
        info (DataFrame): Additional information related to the options data.
        last_px (float): The last closing price of the underlying asset.
        hist_vol (float): The historical volatility of the underlying asset.
        rate_curve (RateCurve): An object representing the interest rate curve for the as-of date.
        clean_data (DataFrame): The cleaned and processed options data with additional calculated fields.
    """
    def __init__(self, as_of_date = None, path = r'./spx_quotes.csv'):

        self.path = path
        self.options_data, self.info = self.get_option_data()
        
        if as_of_date == None: 
            self.as_of_date = pd.to_datetime(self.info.loc['Date']).iloc[0]
            self.str_as_of_date = dt.strftime(self.as_of_date, '%Y-%m-%d')
            self.last_px = self.info.loc['Last'].iloc[0]
        else:
            self.str_as_of_date = as_of_date
            self.as_of_date = dt.strptime(as_of_date, '%Y-%m-%d')
            self.last_px = get_aod_close(dt.strftime(self.as_of_date,'%Y-%m-%d'))

        self.options_data = self.options_data.loc[self.options_data.index > self.as_of_date]
        self.hist_vol = get_hist_vol(self.as_of_date)
        self.rate_curve = rateCurve(self.str_as_of_date)
        self.clean_data = self.get_clean_data()


    def get_option_data(self):
        """
        Retrieves and processes option data from a CSV file.

        This function reads option data from a specified file path, processes the data to calculate
        mid-prices for bid and ask, and returns the processed data along with additional information.

        Returns:
            tuple: A tuple containing two DataFrames:
                - df_options: The processed options data with calculated mid-prices.
                - df_info: Additional information related to the options data.
        """
        df_info, df_options = read_options_csv(self.path)
        df_options.index = pd.to_datetime(df_options.index)
        
        df_options.insert(
            df_options.columns.get_loc('Bid') + 1,
            'Mid',
            df_options[['Bid','Ask']].median(axis = 1))

        df_options.insert(
            df_options.columns.get_loc('Bid.1') + 1,
            'Mid.1',
            df_options[['Bid.1','Ask.1']].median(axis = 1))
        
        return df_options, df_info
    

    def get_clean_data(self):
        """
        Cleans and prepares option data by removing unnecessary columns and adding calculated fields.

        This function processes the options data by dropping unnecessary columns, adding relevant
        information such as the as-of date, time to maturity, risk-free rate, and implied volatility
        for both calls and puts. It returns the cleaned and enriched DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the cleaned and enriched options data.
        """
        df_clean = self.options_data
        df_clean = df_clean.drop(['Last Sale','Net', 'Bid', 'Ask', 'Volume', 'Open Interest', 'Last Sale.1','Net.1', 'Bid.1', 'Ask.1', 'Volume.1', 'Open Interest.1'], axis=1)
        df_clean.insert(0, 'as_of_date', self.str_as_of_date)
        df_clean.insert(1, 'time_to_maturity', [date_difference_in_years(self.as_of_date, date) for date in df_clean.index])
        df_clean.insert(2, 'risk_free_rate', [self.rate_curve.get_rate(date)[0] for date in df_clean.index])
        df_clean.insert(df_clean.columns.get_loc('Calls') + 1, 'Implied_Vol',
                        [self.get_IV(
                            C_market = df_clean['Mid'].loc[df_clean['Calls'] == call].iloc[0],
                            S= float(self.info.loc['Last'].iloc[0]),
                            K=df_clean['Strike'].loc[df_clean['Calls'] == call].iloc[0],
                            T= df_clean['time_to_maturity'].loc[df_clean['Calls'] == call].iloc[0],
                            r = df_clean['risk_free_rate'].loc[df_clean['Calls'] == call].iloc[0],
                            is_call=True
                            ) for call in df_clean['Calls']])
        df_clean.insert(df_clean.columns.get_loc('Puts') + 1, 'Implied_Vol.1',
                        [self.get_IV(
                            C_market = df_clean['Mid.1'].loc[df_clean['Puts'] == put].iloc[0],
                            S= float(self.info.loc['Last'].iloc[0]),
                            K=df_clean['Strike'].loc[df_clean['Puts'] == put].iloc[0],
                            T= df_clean['time_to_maturity'].loc[df_clean['Puts'] == put].iloc[0],
                            r = df_clean['risk_free_rate'].loc[df_clean['Puts'] == put].iloc[0],
                            is_call=False
                            ) for put in df_clean['Puts']])        
        return df_clean


    def get_BSM_price(self,S, K, T, r, sigma, call =True):
        """
        Calculate the price of a European call or put option using the Black-Scholes-Merton (BSM) model.

        Args:
            S (float): The current price of the underlying asset.
            K (float): The strike price of the option.
            T (float): The time to maturity of the option (in years).
            r (float): The risk-free interest rate (annualized).
            sigma (float): The volatility of the underlying asset.
            call (bool): A flag indicating whether the option is a call (True) or a put (False).

        Returns:
            float: The price of the option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - (sigma * np.sqrt(T))

        if call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


    def get_vega(self, S, K, T, r, sigma):
        """
        Calculate the vega of a European option using the Black-Scholes-Merton (BSM) model.

        Vega represents the sensitivity of the option price to changes in the volatility of the underlying asset.

        Args:
            S (float): The current price of the underlying asset.
            K (float): The strike price of the option.
            T (float): The time to maturity of the option (in years).
            r (float): The risk-free interest rate (annualized).
            sigma (float): The volatility of the underlying asset.

        Returns:
            float: The vega of the option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
        return S * np.sqrt(T) * norm.pdf(d1)    


    def get_IV_NewtonRaphson(self,C_market, S, K, T, r, is_call= True, tol=1e-5, max_iterations=500):
        
        initial_guess= self.hist_vol
        sigma = initial_guess

        for _ in range(max_iterations):
            C_BS = self.get_BSM_price(S, K, T, r, sigma, call= is_call)
            diff = C_BS - C_market
            if abs(diff) < tol:
                return sigma
            if self.get_vega(S, K, T, r, sigma) == 0:
                # raise ValueError(f"Vega is zero, cannot perform division. vars are {self.get_vega(S, K, T, r, sigma), S, K, T, r, sigma}")
                return np.nan
            sigma -= diff / self.get_vega(S, K, T, r, sigma)
        
        return np.nan
        

    def get_IV(self, C_market, S, K, T, r, is_call=True, tol=1e-5, max_iterations=500):
        # Define the function whose root we are trying to find
        def objective_function(sigma):
            return self.get_BSM_price(S, K, T, r, sigma, call=is_call) - C_market

        # Initial guess for volatility
        initial_guess = self.hist_vol
        # Try in class Newton-Raphson Method
        try:
            sigma = self.get_IV_NewtonRaphson(C_market, S, K, T, r, is_call)
            if (sigma != np.nan) and (abs(objective_function(sigma)) < tol):
                return sigma
        except:
            pass
        # Try library Newton-Raphson Method
        try:
            sigma = newton(objective_function, initial_guess, tol=tol, maxiter=max_iterations)
            if abs(objective_function(sigma)) < tol:
                return sigma
        except:
            pass

        # Try Brent's Method
        try:
            
            lower_bound = 0.01  
            upper_bound = 5.0   
            sigma = brentq(objective_function, lower_bound, upper_bound, xtol=tol, maxiter=max_iterations)
            if abs(objective_function(sigma)) < tol:
                return sigma
        except:
            pass

        # Try Secant Method
        try:
            
            sigma = newton(objective_function, initial_guess, fprime=None, args=(), tol=tol, maxiter=max_iterations, x1=initial_guess*1.1)
            if abs(objective_function(sigma)) < tol:
                return sigma
        except:
            pass

        # Try Bisection Method
        try:
            sigma = bisect(objective_function, lower_bound, upper_bound, xtol=tol, maxiter=max_iterations)
            if abs(objective_function(sigma)) < tol:
                return sigma
        except:
            pass

        # print("All methods failed to converge.")
        return np.nan