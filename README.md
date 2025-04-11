# Implied_vol_project

python version : 3.12.4

My code is composed of three modules:
class_options
class_rates
data_toolbox

The module class_options contains the class SPX_Options which creates and object that imports and cleans the options data provided in the csv as well as calculate the implied volatilities for a given as-of date. 

The module class_rates contains the class rateCurve which imports the US rate curve from ustreasurycurve library and the SOFR rate from the NY Fed API for a given as_of_date. the class contains also methods to clean and aggregate the rate and interpolate the rates for any given maturity. The class also contains a method (get_option_implied_rate) that calculates the rate implied by an option price through the Put-Call Parity. Yet, it is not used in this context as access to the relevant risk-free rate curve seems more relevant. 

The module data_toolbox contains several functions that import, clean or calculate particular data and data sets.

Finaly a note book main.ipynb to visualise the data.
