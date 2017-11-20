import os
import h5py
import numpy as np
from sklearn import mixture
from scipy import special


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
    return A0*np.exp(-(x-m0)**2 / (2*s0**2)) + A1*np.exp(-(x-m1)**2 / (2*s1**2))


def is_iterated(func):
    """Evaluates independent variables to see if it is a list(-ish) type.

    Need to have numpy functions defined in the namespace for CsPy, but we dont
    actually want to pollute the namespace.
    """
    from numpy import *
    try:
        tmp = eval(func.value)
    except NameError as e:
        print(e)
        return False
    return (type(tmp) == 'list') | (type(tmp) == ndarray) | (type(tmp) == tuple)


def get_iteration_variables(h5file, iterations):
    """Returns a list of variables that evaluate to a list"""
    i_vars = []
    if iterations > 1:
        for i in h5file['settings/experiment/independentVariables/'].iteritems():
            # how to eval numpy functions withoout namespace
            if is_iterated(i[1]['function']):
                i_vars.append(i[0])
    return i_vars


class QDP:
    """A data processing class for quantizing data.

    The raw data format is the following:
    [ # list of experiments
        { # experiment dict
            experiment_name: `exp_name`,
            source_file: `source file path`
            # variable with lowest index is the innermost loop
            # other variables/settings can be included in the iteration object,
            # but these must be because they are the actual changed variables
            variable_list: [`var_0`, `var_1`],
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

    def get_cuts(self):
        return self.cuts

    def get_thresholds(self, max_atoms=1, hbins=0, save_cuts=True, exp=0, itr=0):
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
            cuts[s] = [intersection(*guess)]
            self.rload[s] = frac(*guess)
            ret_val.append({
                'hist_x': bin_edges[:-1],
                'hist_y': hist,
                'max_atoms': max_atoms,
                'fit_params': guess,
                'fit_cov': guess,
                'cuts': cuts[-1],
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
            exp_data = {
                'experiment_name': os.path.basename(full_filepath),
                'source_file': full_filepath,
                'variable_list': get_iteration_variables(h5file, iterations),
                'iterations': {}
            }
            # step through iterations
            for i in e[1]['iterations/'].iteritems():
                exp_data['iterations'][int(i[0])] = self.process_iteration(i[1])
            exps.append(exp_data)
        return exps

    def process_iteration(self, h5_iter):
        iteration_obj = {
            'variables': {},
            'timeseries_data': [],  # cant be numpy array because of different measurement number
            'signal_data': [],  # cant be numpy array because of different measurement number
        }
        # copy variable values over
        for v in h5_iter['variables'].iteritems():
            iteration_obj['variables'][v[0]] = v[1]
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
        drop_bins = variables['throwaway_bins'].value
        meas_bins = variables['measurement_bins'].value
        tmp = np.array(measurement['data/counter/data'].value)
        ptr = 0
        shots = len(tmp)/(drop_bins + meas_bins)
        timeseries_data = np.zeros((shots, meas_bins))
        for s in range(shots):
            ptr += drop_bins
            timeseries_data[s] = tmp[ptr:ptr + meas_bins]
            ptr += meas_bins
        return timeseries_data
