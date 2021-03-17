from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def format_colname(prefix, suffix):
    return "{:s}_{:s}".format(prefix, suffix)


"""A dictionary to get a date part cardinality given a general name"""
__date_part_cardinality = {
    "MONTH": 12,
    "DAY": 31,
    "DAY_OF_WEEK": 7,
    "HOUR": 24,
    "MINUTE": 60,
    "SECOND": 60
}

"""A dictionary to get a date part extractor given a general name"""
__date_part_funcs = {
    "MONTH": lambda x: x.month,
    "DAY": lambda x: x.day,
    "DAY_OF_WEEK": lambda x: x.dayofweek,
    "HOUR": lambda x: x.hour,
    "MINUTE": lambda x: x.minute,
    "SECOND": lambda x: x.second
}


def date_to_dateparts(df, col_name, parts= list(__date_part_funcs.keys()), new_col_name_prefix=None):
    if new_col_name_prefix is None:
        new_col_name_prefix = col_name
    for part in parts:
        assert part in list(__date_part_funcs.keys()), \
            "part '{}' is not known. Available are {}".format(
                part, ", ".join(list(__date_part_funcs.keys())))
    return pd.DataFrame({
        format_colname(new_col_name_prefix, part):
        df[col_name].apply(__date_part_funcs.get(part))
        for part in parts}, index=df.index)


def date_to_cyclical(df, col_name, parts=list(__date_part_funcs.keys()), new_col_name_prefix=None):
    if new_col_name_prefix is None:
        new_col_name_prefix = col_name
    for part in parts:
        assert part in list(__date_part_funcs.keys()), \
            "part '{}' is not known. Available are {}".format(
                part, ", ".join(list(__date_part_funcs.keys())))
    names = [format_colname(new_col_name_prefix, part) for part in parts]
    names_sin = ["{:s}_SIN".format(name) for name in names]
    names_cos = ["{:s}_COS".format(name) for name in names]
    values = [df[col_name].apply(__date_part_funcs.get(part)) /
              (2.0 * np.pi * __date_part_cardinality.get(part)) for part in parts]
    values_sin = [col.apply(np.sin) for col in values]
    values_cos = [col.apply(np.cos) for col in values]
    result = pd.concat(values_sin + values_cos, axis=1)
    result.columns = names_sin + names_cos
    return result


def to_circular_variable(df, col_name, cardinality):
    return pd.DataFrame({
        # note that np.sin(df[col_name] / float(cardinalilty...)) gives different values, probably rounding
        "{:s}_SIN".format(col_name): df[col_name].apply(lambda x: np.sin(x / float(cardinality * 2 * np.pi))),
        "{:s}_COS".format(col_name): df[col_name].apply(lambda x: np.cos(x / float(cardinality * 2 * np.pi)))
    }, index=df.index)


class DateOneHotEncoding(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms date columns into one hot encoding of the parts (day, hour, ..).
    The original date column will be removed.
    To be used by sklearn pipelines
    """

    def __init__(self, date_columns, parts=list(["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]),
                 new_column_names=None, drop=True):
        """
        :param date_columns: the column names of the date columns to be expanded in one hot encodings
        :param new_column_names: the names to use as prefix for the generated column names
        :param drop: whether or not to drop the original column
        :param parts: the parts to extract from the date columns, and to then transform into one-hot encodings
        """
        self.drop = drop
        self.parts = parts
        if new_column_names is None:
            self.new_column_names = date_columns
        else:
            self.new_column_names = new_column_names
        self.date_columns = date_columns
        self.one_hot_encoding_model = OneHotEncoder(sparse=False, handle_unknown='ignore'
                                                    # , n_values=datepart_maxvalue
                                                    )
        self.encoding_pipeline = Pipeline([
            ('labeler', StringIndexer()),
            ('encoder', self.one_hot_encoding_model)
        ])
        assert (len(self.date_columns) == len(self.new_column_names)), \
            "length of new column names is not equal to given column names"

    def all_to_parts(self, X):
        parts = [date_to_dateparts(X, old_name, self.parts, new_name)
                 for old_name, new_name in zip(self.date_columns, self.new_column_names)]
        result = pd.concat(parts, axis=1, join_axes=[X.index])
        return result

    def fit(self, X, y):
        parts = self.all_to_parts(X)
        self.encoding_pipeline.fit(parts)
        # original column i is mapped to values in range resulting_indices[i] .. resulting_indices[i+1]
        resulting_indices = self.one_hot_encoding_model.feature_indices_
        active_features = self.one_hot_encoding_model.active_features_
        new_names = [''] * (np.max(resulting_indices) + 1)
        for i, item in enumerate(parts.columns):
            for j in range(resulting_indices[i], resulting_indices[i + 1]):
                new_names[j] = "{}-{}".format(item, j)
        self.fitted_names = [new_names[i] for i in active_features]
        return self

    def transform_one_hots(self, X):
        np_frame = self.encoding_pipeline.transform(self.all_to_parts(X))
        return pd.DataFrame(np_frame, columns=self.fitted_names)

    def transform(self, X):
        new_columns = self.transform_one_hots(X)
        old_columns = X.drop(self.date_columns, axis=1,
                             inplace=False) if self.drop else X

        return pd.concat([old_columns, new_columns], axis=1, join_axes=[X.index])


class DateCyclicalEncoding(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms date columns into cyclical numerical columns.
    The original date column will be removed.
    To be used by sklearn pipelines
    """

    def __init__(self, date_columns, parts=list(["DAY", "DAY_OF_WEEK", "HOUR", "MINUTE", "MONTH", "SECOND"]),
                 new_column_names=None, drop=True):
        """
        :param date_columns: the column names of the date columns to be expanded in one hot encodings
        :param new_column_names: the names to use as prefix for the generated column names
        :param drop: whether or not to drop the original column
        :param parts: the parts to extract from the date columns, and to then transform into one-hot encodings
        """
        self.parts = parts
        self.drop = drop
        if new_column_names is None:
            self.new_column_names = date_columns
        else:
            self.new_column_names = new_column_names
        self.date_columns = date_columns
        assert (len(self.date_columns) == len(self.new_column_names))

    def all_to_cyclical_parts(self, X):
        parts = [date_to_cyclical(X, old_name, self.parts, new_name)
                 for old_name, new_name in zip(self.date_columns, self.new_column_names)]
        return pd.concat(parts, axis=1, join_axes=[X.index])

    def fit(self, X, y):
        return self

    def transform(self, X):
        new_columns = self.all_to_cyclical_parts(X)
        old_columns = X.drop(self.date_columns, axis=1,
                             inplace=False) if self.drop else X
        return pd.concat([old_columns, new_columns], axis=1, join_axes=[X.index])


# like sklearn's transformers, but then on pandas DataFrame
class PdLagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag):
        self.lag = lag

    def fit(self, X, y=None, **fit_params):
        return self

    def do_transform(self, dataframe):
        return (dataframe.shift(self.lag)
                .rename(columns=lambda c: "{}_lag{}".format(c, self.lag)))

    def transform(self, X):
        try:
            return self.do_transform(X)
        except AttributeError:
            return self.do_transform(pd.DataFrame(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class PdWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func, **rolling_params):
        self.func = func
        self.rolling_params = rolling_params

    def fit(self, X, y=None, **fit_params):
        return self

    def do_transform(self, dataframe):
        return (self.func(dataframe.rolling(**self.rolling_params))
                .rename(columns=lambda c: "{}_{}".format(c, "".join(
                    ["{}{}".format(k, v) for k, v in self.rolling_params.items()]))))

    def transform(self, X):
        try:
            return self.do_transform(X)
        except AttributeError:
            return self.do_transform(pd.DataFrame(X))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)



class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms a high-capacity categorical value
    into Weigh of Evidence scores. Can be used in sklearn pipelines.
    """

    def __init__(self, verbose=0, cols=None, return_df=True,
                 smooth=0.5, fillna=0, dependent_variable_values=None):
        """
        :param smooth: value for additive smoothing, to prevent divide by zero
        """
        # make sure cols is a list of strings
        if not isinstance(cols, list):
            cols = [cols]

        self.stat = {}
        self.return_df = return_df
        self.verbose = verbose
        self.cols = cols
        self.smooth = smooth
        self.fillna = fillna
        self.dependent_variable_values = dependent_variable_values

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                'Input should be an instance of pandas.DataFrame()')

        if self.dependent_variable_values is not None:
            y = self.dependent_variable_values

        df = X[self.cols].copy()
        y_col_index = len(df.columns) + 1
        df[y_col_index] = np.array(y)

        def get_totals(x):
            total = np.size(x)
            pos = max(float(np.sum(x)), self.smooth)
            neg = max(float(total - pos), self.smooth)
            return pos, neg

        # get the totals per class
        total_positive, total_negative = get_totals(y)
        if self.verbose:
            print("total positives {:.0f}, total negatives {:.0f}".format(
                total_positive, total_negative))

        def compute_bucket_woe(x):
            bucket_positive, bucket_negative = get_totals(x)
            return np.log(bucket_positive / bucket_negative)

        # compute WoE scores per bucket (category)
        stat = {}
        for col in self.cols:

            if self.verbose:
                print(
                    "computing weight of evidence for column {:s}".format(col))

            stat[col] = ((df.groupby(col)[y_col_index].agg(compute_bucket_woe)
                          + np.log(total_negative / total_positive)).to_dict())

        self.stat = stat

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                'Input should be an instance of pandas.DataFrame()')

        df = X.copy()

        # join the WoE stats with the data
        for col in self.cols:

            if self.verbose:
                print("transforming categorical column {:s}".format(col))

            stat = pd.DataFrame.from_dict(self.stat[col], orient='index')

            ser = (pd.merge(df, stat, left_on=col, right_index=True, how='left')
                   .sort_index()
                   .reindex(df.index))[0]

            # fill missing values with
            if self.verbose:
                print("{:.0f} NaNs in transformed data".format(
                    ser.isnull().sum()))
                print("{:.4f} mean weight of evidence".format(ser.mean()))

            df[col] = np.array(ser.fillna(self.fillna))

        if not self.return_df:
            out = np.array(df)
        else:
            out = df

        return out


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except:
            print("Could not find selected columns {:s} in available columns {:s}".format(
                self.columns, X.columns))
            raise


class StringIndexer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dictionaries = dict()
        self.columns = list()

    def fit(self, X, y=None):
        self.columns = X.columns.values
        for col in self.columns:
            categories = np.unique(X[col])
            self.dictionaries[col] = dict(
                zip(categories, range(len(categories))))
        return self

    def transform(self, X):
        column_array = []
        for col in self.columns:
            dictionary = self.dictionaries[col]
            na_value = len(dictionary) + 1
            transformed_column = X[col].apply(
                lambda x: dictionary.get(x, na_value))
            column_array.append(transformed_column.values.reshape(-1, 1))
        return np.hstack(column_array)


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """
    Leave one out transformation for high-capacity categorical variables.
    """

    def __init__(self, with_stdevs=True):

        self.with_stdevs = with_stdevs
        self.means = {}
        self.stdevs = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.means.keys():

            mean_col_name = "{:s}_MEAN".format(col)
            df[mean_col_name] = df.merge(pd.DataFrame(self.means[col]),
                                         how='left', left_on=[col], right_index=True)['y']
            if self.with_stdevs:
                std_col_name = "{:s}_STD".format(col)
                df[std_col_name] = df.merge(pd.DataFrame(self.stdevs[col]),
                                            how='left', left_on=[col], right_index=True)['y']

            df.drop(col, axis=1, inplace=True)

        return df

    def fit_transform(self, X, y):
        """will be used during pipeline fit"""
        df = X.copy()
        df['y'] = y
        for col in df.columns.difference(['y']):

            mean_col_name = "{:s}_MEAN".format(col)

            grouped = df.groupby(col)['y']

            self.means[col] = grouped.mean()
            df[mean_col_name] = grouped.transform(self._loo_means)

            if self.with_stdevs:
                std_col_name = "{:s}_STD".format(col)
                self.stdevs[col] = grouped.std()
                df[std_col_name] = grouped.transform(self._loo_stdevs)

            df.drop(col, axis=1, inplace=True)

        df.drop('y', axis=1, inplace=True)
        return df

    def _loo_means(self, s):
        n = len(s)
        loo_means = (s.sum() - s) / (n - 1)
        return loo_means * np.random.normal(loc=1.0, scale=0.01, size=n)

    def _loo_stdevs(self, s):
        n = len(s)
        if n > 1:
            loo_means = self._loo_means(s)
            sum_of_sq = n * s.std() ** 2
            loo_stdevs = np.sqrt(
                abs((sum_of_sq - (s - s.mean()) * (s - loo_means))) / (n - 1))
        else:
            loo_stdevs = np.array([0])

        return loo_stdevs * np.random.normal(loc=1.0, scale=0.01, size=n)


class FeatureRatioCal(BaseEstimator, TransformerMixin):
    def __init__(self, columns):  # no *args or **kargs
        self.columns = columns

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        ratio = X[self.columns[0]] / (X[self.columns[1]] + 1e-6)
        ratio_name = '_'.join([self.columns[0], self.columns[1], 'ratio'])
        X.loc[:, ratio_name] = ratio
        return X

class FeatureDiffCal(BaseEstimator, TransformerMixin):
    def __init__(self, columns):  # no *args or **kargs
        self.columns = columns

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        ratio = X[self.columns[0]] - X[self.columns[1]]
        ratio_name = '_'.join([self.columns[0], self.columns[1], 'sub'])
        X.loc[:, ratio_name] = ratio
        return X
