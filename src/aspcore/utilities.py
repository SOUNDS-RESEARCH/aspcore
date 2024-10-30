import copy
import numpy as np

import aspcore.filter as fc


def power_of_filtered_signal(src, ir, num_samples):
    """Returns an estimate of average power of the signal after filtered through an impulse response
        
    Requires non-standard dependency aspcore

    Parameters
    ----------
    src : source object
        The source generating the signal
        Can be any object with a get_samples method that takes an integer num_samples as argument
        and returns an ndarray of shape (num_channels, num_samples)
    ir : ndarray of shape (num_channels, num_recievers, ir_len)
        The impulse response to filter the signal with
    num_samples : int
        The number of samples to use for the estimate. If the signal is periodic, this should be the period length
        
    Returns
    -------
    avg_pow : ndarray of shape (num_recievers,)
        The average power for each receiver channel. Will only have non-negative values.
    """
    assert ir.ndim == 3
    ir_len = ir.shape[-1]
    src_copy = copy.deepcopy(src)
    in_sig = src_copy.get_samples(num_samples+ir_len-1)

    filt = fc.create_filter(ir)
    filt_sig = filt.process(in_sig)
    filt_sig = filt_sig[...,ir_len-1:]
    avg_pow = np.mean(filt_sig**2, axis=-1)
    return avg_pow

def is_power_of_2(x):
    """Returns True if x is a power of 2, False otherwise

    Parameters
    ----------
    x : int
        The number to check

    Returns
    -------
    is_power_of_2 : bool
        True if x is a power of 2, False otherwise
    """
    return is_integer(np.log2(x))

def next_power_of_two(x):
    """ Returns the smallest number that is both a power of two and larger or equal to x. 
    """
    return int(2**(np.ceil(np.log2(x))))

def is_integer(x):
    """Returns True if x is an integer, False otherwise
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if np.all(np.isclose(x, x.astype(int))):
        return True
    return False



def get_smallest_coprime(N):
    """Get the smallest value that is coprime with N
    
    Parameters
    ----------
    N : int
        The number to find a coprime to
    
    Returns
    -------
    coprime : int
        The smallest coprime to N
    """
    assert N > 2 #don't have to deal with 1 and 2 at this point
    for i in range(2,N):
        if np.gcd(i,N):
            return i

def next_divisible(divisor, min_value):
    """Gives the smallest integer divisible by divisor, that is strictly larger than min_value
    
    Parameters
    ----------
    divisor : int
        The number to be divisible by
    min_value : int
        The smallest acceptable value

    Returns
    -------
    next_divisible_number   : int
        The smallest integer satisfying the conditions
    """
    rem = (min_value + divisor) % divisor
    return min_value + divisor - rem



def simplify_ratio(a : int, b : int):
    """Simplifies the ratio a/b into the simplest possible ratio where both numerator and denominator are integers

    Parameters
    ----------
    a : int
        numerator
    b : int
        denominator

    Returns
    -------
    a : int
        simplified numerator
    b : int
        simplified denominator
    """
    d = np.gcd(a,b)
    while d != 1:
        a = a // d
        b = b // d
        d = np.gcd(a,b)
    return a,b




def block_process_idxs(num_samples : int, block_size : int, overlap : int, start_idx=0):
    """Yields the starting index for each block, for block processing a signal

    Parameters
    ----------
    num_samples : int
        total number of samples for the signal that should be processed
    block_size : int
        the size of each of the blocks
    overlap : int
        the amount each block should be overlapped at the output
    start_idx : int
        can be supplied if the processing should start at another place
        of the original signal than idx = 0

    Yields
    -------
    idx : int
        can be used to get your block as signal[..., idx:idx + block_size]
    """
    assert 0 <= overlap < block_size
    assert 0 <= start_idx < num_samples
    hop = block_size - overlap
    #left_in_block = block_size - start_idx

    #indices = []
    
    sample_counter = start_idx
    while sample_counter+block_size < num_samples:
        #block_len = min(num_samples - sample_counter, left_in_block)
        yield sample_counter 
        #indices.append(sample_counter)


        sample_counter += hop






class PhaseCounter:
    """
    An index counter to keep track of non-overlapping continous phases
    
    Example:
    A processor needs the first 2000 samples for an initialization, 
    then must wait 5000 samples before beginning the real processing step.
    The class can then be used by providing
    phase_def = {
    'init' : 2000,
    'wait' : 5000,
    'process' : np.inf
    }
    and then checking if phase_counter.phase == 'init'
    or if phase_counter.current_phase_is('init'):
    
    The number is how many samples that each phase should be
    The first phase will start at sample 0.

    np.inf represents an infinite length
    This should naturally only be used for the last phase
    If all phases has finished, the phase will be None. 

    first_sample will be True on the first sample of each phase,
    allowing running one-time functions in each phase

    Extended implementation to blocksize != 1 can be done later
    """
    def __init__(self, phase_lengths, verbose=False):
        assert isinstance(phase_lengths, dict)
        self.phase_lengths = phase_lengths
        self.verbose = verbose
        self.phase = None
        self.first_sample = True
        

        #phase_lengths = {name : length for name, length in self.phase_lengths.items() if length != 0}
        #phase_lengths = {name : length for name, length in self.phase_lengths.items()}
        
        #phase_idxs = [i for i in self.phase_lengths.values() if i != 0]
        self.phase_lengths = {name : i if i >= 0 else np.inf for name, i in self.phase_lengths.items()}
        #assert all([i != 0 for i in p_len])
        self.start_idxs = np.cumsum(list(self.phase_lengths.values())).tolist()
        self.start_idxs = [i if np.isinf(i) else int(i) for i in self.start_idxs]
        self.start_idxs.insert(0,0)

        self.phase_names = list(self.phase_lengths.keys())
        if self.start_idxs[-1] < np.inf:
            self.phase_names.append(None)
        else:
            self.start_idxs.pop()

        self.start_idxs = {phase_name:start_idx for phase_name, start_idx in zip(self.phase_names, self.start_idxs)}

        self._phase_names = [phase_name for phase_name, phase_len in self.phase_lengths.items() if phase_len > 0]
        self._start_idxs = [start_idx for start_idx, phase_len in zip(self.start_idxs.values(), self.phase_lengths.values()) if phase_len > 0]

        self.idx = 0
        self.next_phase()

    def next_phase(self):
        if self.verbose:
            print(f"Changed phase from {self.phase}")
            
        self.phase = self._phase_names.pop(0)
        self._start_idxs.pop(0)
        if len(self._start_idxs) == 0:
            self._start_idxs.append(np.inf)
        self.first_sample = True
        
        if self.verbose:
            print(f"to {self.phase}")

    def progress(self):
        self.idx += 1
        if self.idx >= self._start_idxs[0]:
            self.next_phase()
        else:
            self.first_sample = False

    def current_phase_is(self, phase_name):
        return self.phase == phase_name



class EventCounter:
    """
    An index counter to keep track of events that should 
    only happen every x samples

    event_def is a dictionary with all event
    each entry is 'event_name' : (frequency, offset)

    Example:
    event_counter = EventCounter({'event_1' : (256,0), 'event_2' : (1,0), 'event_3' : (1024,256)})
    event_2 will happen every sample, event_1 every 256 samples
    First at sample 256 all three events will happen simultaneouly. 

    To be used as:
    if 'event_name' in event_counter.event:
    do_thing()

    """
    def __init__(self, event_def):
        self.event_def = event_def
        self.event = []

        self.freq = {name : freq for name, (freq, offset) in event_def.items()}
        self.offset = {name : offset for name, (freq, offset) in event_def.items()}

        self.idx = 0

    def add_event(self, name, freq, offset):
        self.event_def[name] = (freq, offset)

    def check_events(self):
        self.event = []
        for name, (freq, offset) in self.event_def.items():
            if (self.idx - offset) % freq == 0:
                self.event.append(name)

    def progress(self):
        self.idx += 1 
        self.check_events()
