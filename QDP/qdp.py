import os
import h5py
import numpy as np
from sklearn import mixture
from scipy import special
from scipy import optimize
import subprocess
import json
import ivar


def intersection(A0, A1, m0, m1, s0, s1):
    return (m1*s0**2-m0*s1**2-np.sqrt(s0**2*s1**2*(m0**2-2*m0*m1+m1**2+2*np.log(A0/A1)*(s1**2-s0**2))))/(s0**2-s1**2)


def area(A0, A1, m0, m1, s0, s1):
    return np.sqrt(np.pi/2)*(A0*s0+A0*s0*special.erf(m0/np.sqrt(2)/s0)+A1*s1+A1*s1*special.erf(m1/np.sqrt(2)/s1))


# Normed Overlap for arbitrary cut point
def overlap(xc, A0, A1, m0, m1, s0, s1):
    err0 = A0*np.sqrt(np.pi/2)*s0*(1-special.erf((xc-m0)/np.sqrt(2)/s0))
    err1 = A1*np.sqrt(np.pi/2)*s1*(special.erf((xc-m1)/np.sqrt(2)/s1)+special.erf(m1/np.sqrt(2)/s1))
    return (err0+err1)/area(A0, A1, m0, m1, s0, s1)


# Relative Fraction in 1
def frac(A0, A1, m0, m1, s0, s1):
    return 1/(1+A0*s0*(1+special.erf(m0/np.sqrt(2)/s0))/A1/s1/(1+special.erf(m1/np.sqrt(2)/s1)))


def dblgauss(x, A0, A1, m0, m1, s0, s1):
    return A0*np.exp(-(x-m0)**2 / (2*s0**2))/np.sqrt(2*np.pi*s0**2) + A1*np.exp(-(x-m1)**2 / (2*s1**2))/np.sqrt(2*np.pi*s1**2)


def get_iteration_variables(h5file, iterations):
    """Returns a list of variables that evaluate to a list"""
    i_vars = []
    i_vars_desc = {}
    if iterations > 1:
        for i in h5file['settings/experiment/independentVariables/'].iteritems():
            # how to eval numpy functions withoout namespace
            if ivar.is_iterated(i[1]['function']):
                i_vars.append(i[0])
                i_vars_desc[i[0]] = {
                    'description': i[1]['description'][()],
                    'function': i[1]['function'][()],
                }
    return (i_vars, i_vars_desc)


def binomial_error(ns, n):
    """Normal approximation interval, see: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval"""
    alpha = 1-0.682
    z = 1-0.5*alpha
    errs = np.zeros_like(ns)
    for i in xrange(len(ns)):
        # if ns is 0 or n, then use ns = 0.5 or n - 0.5 for the error calculation so we dont get error = 0
        if int(ns[i]) == 0:
            ns[i] = 0.5
        if int(ns[i]) == int(n):
            ns[i] = n - 0.5
        if n == 0:
            print "no loading observed"
            errs[i] = np.nan
        else:
            errs[i] = (z/float(n))*np.sqrt(ns[i]*(1.0-float(ns[i])/float(n)))
    return errs


def jsonify(data):
    """Prep for serialization."""
    json_data = dict()
    for key, value in data.iteritems():
        if isinstance(value, list):  # for lists
            value = [jsonify(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):  # for nested lists
            value = jsonify(value)
        if isinstance(key, int):  # if key is integer: > to string
            key = str(key)
        if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json_data


class QDP:
    """A data processing class for quantizing data.

    cuts format is:
    [
        [0-1 threshold, 1-2 threshold, ... (n-1)-n threshold],  # shot 0
        ...  # shot 1
    ]

    The raw data format is the following:
    [ # list of experiments
        { # experiment dict
            experiment_name: `exp_name`,
            source_file: `source file path`
            # variable with lowest index is the innermost loop
            # other variables/settings can be included in the iteration object,
            # but these must be because they are the actual changed variables
            variable_list: [`var_0`, `var_1`],
            variable_desc: {
                `var_0`: {'description': `description`, 'function': `function`},
                `var_1`: {'description': `description`, 'function': `function`},
            }
            iterations: {
                `iter_key`: {
                    timeseries_data: [ # if camera data time series is length 1
                            [shot_0_timeseries], ... [shot_n-1_timeseries],
                            ...
                    ],
                    signal_data: [ # measurement list
                            [shot_0_signal, ... shot_n-1_signal],
                            ...
                    ],
                    quantized_data: [ # 0, 1, 2, ...
                            [shot_0_quant, ... shot_n-1_quant],
                            ...
                    ],
                    loading: `loading`,
                    retention: `retention`,
                    retention_err: `retention_err`,
                    loaded: `loaded`,
                    variables: {
                        `var_0`: var_0_iter_val,
                        `var_1`: var_1_iter_val,
                    }
                },
                `iter_key`: {
                    timeseries_data: [ # measurement list
                            [shot_0_timeseries], ... [shot_n-1_timeseries],
                            ...
                    ],
                    signal_data: [ # measurement list
                            [shot_0_signal, ... shot_n-1_signal],
                            ...
                    ],
                    quantized_data: [ # 0, 1, 2, ...
                            [shot_0_quant, ... shot_n-1_quant],
                            ...
                    ],
                    loading: `loading`,
                    retention: `retention`,
                    retention_err: `retention_err`,
                    loaded: `loaded`,
                    variables: {
                        `var_0`: var_0_iter_val,
                        `var_1`: var_1_iter_val,
                    }
                }
            }
        },
        ...
    ]
    """
    def __init__(self, base_data_path=""):
        self.experiments = []
        self.cuts = []
        self.rload = []
        # set a data path to search from
        self.base_data_path = base_data_path
        # save current git hash
        self.version = subprocess.check_output(['git', 'describe', '--always']).strip()

    def apply_thresholds(self, cuts=None, exp='all', loading_shot=0):
        """Digitize data with existing thresholds (default) or with supplied thresholds.

        digitization bins are right open, i.e. the condition  for x in bin i b[i-1] <= x < b[i]
        """
        if cuts is None:
            cuts = self.cuts
        # apply cuts to one or all experiments
        if exp == 'all':
            exps = self.experiments
        else:
            exps = self.experiments[exp]
        for e in exps:
            for i in e['iterations']:
                shots = len(e['iterations'][i]['signal_data'][0])
                # digitize the data
                measurements = len(e['iterations'][i]['signal_data'])
                quant = np.empty((shots, measurements))
                for s in range(shots):
                    # first bin is bin 1
                    quant[s] = np.digitize(e['iterations'][i]['signal_data'][:, s], cuts[s])
                e['iterations'][i]['quantized_data'] = quant.transpose()
                # calculate loading and retention for each shot
                retention = np.empty((shots, ))
                for s in range(shots):
                    retention[s] = np.sum(np.logical_and(
                        quant[loading_shot],
                        quant[s]
                    ), axis=0)
                loading = np.mean(retention[loading_shot])
                retention[loading_shot] = 0.0
                loaded = np.sum(loading)
                e['iterations'][i]['loading'] = loaded/measurements
                e['iterations'][i]['retention'] = retention/loaded
                e['iterations'][i]['retention_err'] = binomial_error(retention, loaded)
                e['iterations'][i]['loaded'] = loaded

        return self.get_retention()

    def get_retention(self, shot=1, fmt='dict'):
        retention = np.empty((
            len(self.experiments),
            len(self.experiments[0]['iterations'].items())
        ))
        err = np.empty_like(retention)
        ivar = np.empty_like(retention)
        loading = np.empty_like(retention)
        for e, exp in enumerate(self.experiments):
            if len(exp['variable_list']) > 1:
                raise NotImplementedError
            if len(exp['variable_list']) == 1:
                ivar_name = exp['variable_list'][0]
            else:
                ivar_name = None
            for i in exp['iterations']:
                retention[e, i] = exp['iterations'][i]['retention'][shot]
                loading[e, i] = exp['iterations'][i]['loading'][()]
                err[e, i] = exp['iterations'][i]['retention_err'][shot]
                if ivar_name is not None:
                    ivar[e, i] = exp['iterations'][i]['variables'][ivar_name][()]
                else:
                    ivar[e, i] = 0
        # if numpy format is requested return it
        if fmt == 'numpy' or fmt == 'np':
            return np.array([ivar, retention, err, loading])
        else:
            # if unrecognized return dict format
            return {
                'retention': retention,
                'loading': loading,
                'error': err,
                'ivar': ivar,
            }

    def get_thresholds(self):
        return self.cuts

    def generate_thresholds(self, max_atoms=1, hbins=0, save_cuts=True, exp=0, itr=0):
        """Find the optimal thresholds for quantization of the data."""
        if max_atoms > 1:
            raise NotImplementedError
        shots = len(self.experiments[exp]['iterations'][itr]['signal_data'][0])
        # use a gaussian mixture model to find initial guess at signal distributions
        gmix = mixture.GaussianMixture(n_components=max_atoms+1)
        ret_val = []
        cuts = []
        self.rload = np.zeros(shots)
        for s in range(shots):
            cuts.append([np.nan]*max_atoms)
            shot_data = self.experiments[exp]['iterations'][itr]['signal_data'][:, s]
            gmix.fit(np.array([shot_data]).transpose())
            # order the components by the size of the signal
            indicies = np.argsort(gmix.means_.flatten())
            guess = []
            for n in range(max_atoms+1):
                idx = indicies[n]
                guess.append([
                    gmix.weights_[idx],  # amplitudes
                    gmix.means_.flatten()[idx],  # x0s
                    np.sqrt(gmix.means_.flatten()[idx])  # sigmas
                ])
            # reorder the parameters
            guess = np.transpose(guess).flatten()
            # bin the data, default binning is just range([0,max])
            if hbins < 1:
                hbins = range(int(np.max(shot_data))+1)
            hist, bin_edges = np.histogram(shot_data, bins=hbins, normed=True)
            try:
                popt, pcov = optimize.curve_fit(dblgauss, bin_edges[:-1], hist, guess)
                cuts[s] = [intersection(*popt)]
                self.rload[s] = frac(*popt)
            except RuntimeError:
                cuts[s] = np.nan  # [intersection(*guess)]
                self.rload[s] = np.nan  # frac(*guess)
            ret_val.append({
                'hist_x': bin_edges[:-1],
                'hist_y': hist,
                'max_atoms': max_atoms,
                'fit_params': popt,
                'fit_cov': pcov,
                'cuts': cuts[-1],
                'guess': guess
            })
        self.cuts = cuts
        return ret_val

    def get_settings_from_file(self, h5_settings):
        """Extract any necessary parameters from the file."""
        pass

    def load_data_file(self, filepath):
        """Load a data file into the class.

        Additional experiments will be appended to an existing list.
        """
        full_path = os.path.join(self.base_data_path, filepath)
        new_experiments = self.load_hdf5_file(full_path)
        self.experiments += new_experiments

    def load_exp_list(self, exps):
        """manually load an experiment list into the class for analysis.

        Probably only useful for testing.
        """
        self.experiments = exps

    def load_hdf5_file(self, full_filepath, h5file=None):
        """Load experiments from data file into standard format.

        returns list of experiment objects
        """
        if h5file is None:
            h5file = h5py.File(full_filepath)
        # load necessary settings
        self.get_settings_from_file(h5file['settings/'])
        h5_exps = h5file['experiments/'].iteritems()
        exps = []
        # step through experiments
        for e in h5_exps:
            # iterations are the same for all experiments in the same file
            # iteration variables are the same for all experiments in a data file
            iterations = len(e[1]['iterations/'].items())
            ivars, ivar_desc = get_iteration_variables(h5file, iterations)
            exp_data = {
                'experiment_name': os.path.basename(full_filepath),
                'source_file': full_filepath,
                'source_filename': os.path.basename(full_filepath),
                'source_path': os.path.dirname(full_filepath),
                'variable_list': ivars,
                'variable_desc': ivar_desc,
                'iterations': {}
            }
            # step through iterations
            for i in e[1]['iterations/'].iteritems():
                exp_data['iterations'][int(i[0])] = self.process_iteration(i[1])
            exps.append(exp_data)
        h5file.close()
        return exps

    def process_iteration(self, h5_iter):
        iteration_obj = {
            'variables': {},
            'timeseries_data': [],  # cant be numpy array because of different measurement number
            'signal_data': [],  # cant be numpy array because of different measurement number
        }
        # copy variable values over
        for v in h5_iter['variables'].iteritems():
            iteration_obj['variables'][v[0]] = v[1][()]
        # copy measurement values over
        for m in h5_iter['measurements/'].iteritems():
            timeseries_data = self.process_measurement(m[1], iteration_obj['variables'])
            iteration_obj['timeseries_data'].append(timeseries_data)
            signal_data = np.sum(timeseries_data, axis=1)
            iteration_obj['signal_data'].append(signal_data)
        # cast as numpy arrays
        iteration_obj['signal_data'] = np.array(iteration_obj['signal_data'])
        iteration_obj['timeseries_data'] = np.array(iteration_obj['timeseries_data'])
        return iteration_obj

    def process_measurement(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of timeseries_data for each shot.
        """
        return self.process_raw_counter_data(measurement, variables)

    def process_raw_counter_data(self, measurement, variables):
        """Retrieve data from hdf5 measurement obj.

        returns numpy array of timeseries_data for each shot.
        """
        drop_bins = variables['throwaway_bins']
        meas_bins = variables['measurement_bins']
        tmp = measurement['data/counter/data'].value.flatten()
        ptr = 0
        shots = tmp.shape[0]/(drop_bins + meas_bins)
        timeseries_data = np.zeros((shots, meas_bins))
        for s in range(shots):
            ptr += drop_bins
            timeseries_data[s] = tmp[ptr:ptr + meas_bins]
            ptr += meas_bins
        return timeseries_data

    def save_experiment_data(self, filename_prefix='data', path=None):
        """Saves data to files with the specified prefix."""
        if path is None:
            path = self.experiments[0]['source_path']
        self.save_json_data(filename_prefix=filename_prefix, path=path)
        self.save_retention_data(filename_prefix=filename_prefix, path=path)

    def save_json_data(self, filename_prefix='data', path=None):
        if path is None:
            path = self.experiments[0]['source_path']
        with open(os.path.join(path, filename_prefix + '.json'), 'w') as f:
            json.dump({
                    'data': map(jsonify, self.experiments),
                    'metadata': {'version': self.version},
                },
                f
            )

    def save_retention_data(self, filename_prefix='data', path=None, shot=1):
        if path is None:
            path = self.experiments[0]['source_path']
        try:
            np.save(
                os.path.join(path, filename_prefix + '.npy'),
                self.get_retention(shot=shot, fmt='numpy'),
                allow_pickle=False
            )
        except KeyError:
            print('Retention data has not been processed.  Not saving.')

    def set_thresholds(self, cuts):
        self.cuts = cuts
