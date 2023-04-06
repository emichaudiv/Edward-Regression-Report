import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from env import user, password, host
import warnings
warnings.filterwarnings('ignore')

def acquire_zillow():
    if os.path.exists('zillow_2017.csv'):
        print('local version found!')
        return pd.read_csv('zillow_2017.csv', index_col=0)
    else:
        ''' Acquire data from Zillow using env imports and rename columns'''

        url = f"mysql+pymysql://{user}:{password}@{host}/zillow"

        query = """
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
        FROM properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusedesc IN ("Single Family Residential",                       
                                      "Inferred Single Family Residential")"""

        df = pd.read_sql(query, url)


        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                                  'bathroomcnt':'bathrooms', 
                                  'calculatedfinishedsquarefeet':'area',
                                  'taxvaluedollarcnt':'tax_value', 
                                  'yearbuilt':'year_built',})
        return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75]) 
        
        iqr = q3 - q1   
        
        upper_bound = q3 + k * iqr   
        lower_bound = q1 - k * iqr   

        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        plot_number = i + 1 

        plt.subplot(1, len(cols), plot_number)

        plt.title(col)

        df[col].hist(bins=5)

        plt.grid(False)

        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        plot_number = i + 1 

        plt.subplot(1, len(cols), plot_number)

        plt.title(col)

        sns.boxplot(data=df[[col]])

        plt.grid(False)

        plt.tight_layout()

    plt.show()

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount'])

    get_hist(df)
    get_box(df)

    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)

    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])   
    return train, validate, test      

def wrangle_zillow(target):
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols
 
    
    
def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df



def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test