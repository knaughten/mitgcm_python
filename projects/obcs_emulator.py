# An emulator to model drift in domain volume over time, and to test
# different OBCS correction strategies to counteract the drift
# continually throughout the simulation.

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from plot_1d import timeseries_multi_plot


# Build a trend for the emulator.
def build_trend (trend, num_time):
    
    if trend == 'linear':
        # Standard deviation of linear trend coefficients (m/y)
        trend_std = 1
    elif trend == 'quadratic':
        # Standard deviation of quadratic trend coefficients (m/y^2)
        trend_std = 0.02
    elif trend == 'sinusoid':
        # Standard deviation of sinusoidal amplitude (m)
        amp_std = 20
        # Mean and standard deviation of sinusoidal period (years)
        per_mean = 60
        per_std = 20

    time = np.arange(num_time + 1)
    if trend == 'linear':
        coeff1 = trend_std*np.random.randn()
        print(('Linear trend of ' + str(coeff1) + ' m/y'))
        trend_full = coeff1*time
    elif trend == 'quadratic':
        coeff1 = trend_std*np.random.randn()
        print(('Quadratic trend of ' + str(coeff1) + 'm/y^2'))
        trend_full = coeff1*time**2
    elif trend == 'sinusoid':
        amplitude = amp_std*np.random.randn()
        period = per_std*np.random.randn() + per_mean
        print(('Sinusoidal trend with amplitude ' + str(amplitude) + ' m and period ' + str(period) + ' years'))
        trend_full = amplitude*np.sin(time*2*np.pi/period)
    trend_steps = np.diff(trend_full)
    return trend_steps


# Run the actual emulator
def run_emulator (trend, feedback, correction, coeff, power, num_time, prob_new_trend=0, trend_steps=None):
    
    # Standard deviation of random walk steps (m/y)
    step_std = 2
    if feedback != 'none':
        # Standard deviation of feedback factor (dimensionless, will be taken as exponent)
        feedback_std = 1.5

    # First build random walk
    random_steps = step_std*np.random.randn(num_time)
    if trend_steps is None:
        # Trend is not precomputed, so build it
        trend_steps = build_trend(trend, num_time)
    # Combine them
    eta_orig = np.cumsum(trend_steps + random_steps)

    # Now build corrected timeseries
    correct_step = 0
    eta_correct = np.empty(num_time)
    for t in range(num_time):
        # Apply correction from previous year (0 on first step)
        if t == 0:
            eta_prev = 0
        else:
            eta_prev = eta_correct[t-1]
        if feedback == 'none':
            # Trend independent of correction
            trend_step_curr = trend_steps[t]            
        else:
            # Select random feedback factor
            ffactor = feedback_std*np.random.randn()
            if feedback == 'positive':
                # Correction amplifies trend (so might be ineffective)
                trend_step_curr = np.exp(np.abs(ffactor))*trend_steps[t]
            elif feedback == 'negative':
                # Correction dampens trend (so might overshoot)
                trend_step_curr = np.exp(-1*np.abs(ffactor))*trend_steps[t]
        eta_correct[t] = eta_prev + correct_step + trend_step_curr + random_steps[t]
        # Calculate correction for next year        
        if correction == 'linear':
            correct_step = -coeff*eta_correct[t]
        elif correction == 'power':
            # Scale coefficient so that effective linear coefficient is 1 for a SSH equal to the standard deviation of random walk steps
            correct_step = -np.sign(eta_correct[t])*np.abs(eta_correct[t])**power/step_std**(power-1)
        if prob_new_trend > 0:
            # Possibility that the trend will change
            if np.random.rand() < prob_new_trend:
                # Recompute the trend steps with new random coefficients
                trend_steps = build_trend()

    # Return results
    return eta_orig, eta_correct


# Run and plot for a single correction method
def run_plot_emulator_single (trend='linear', feedback='none', correction='linear', coeff=1, power=1, prob_new_trend=0, num_time=150):

    eta_orig, eta_correct = run_emulator(trend, feedback, correction, coeff, power, num_time, prob_new_trend=prob_new_trend)
    # Plot results
    timeseries_multi_plot(np.arange(num_time)+1, [eta_orig, eta_correct], ['Original', 'Corrected'], ['blue', 'black'], title='Average sea surface height (emulated)', units='m', dates=False)


# Run and plot for each correction method
def run_plot_emulator_all (trend='linear', feedback='none', coeff=1, power=1.5, num_time=150):

    # Build trend
    trend_steps = build_trend(trend, num_time)
    correction_types = ['linear', 'power']
    correction_colours = ['black', 'green']
    eta_correct = []
    for correction in correction_types:
        eta_orig, eta_correct_tmp = run_emulator(trend, feedback, correction, coeff, power, num_time, trend_steps=trend_steps)
        eta_correct.append(eta_correct_tmp)
    timeseries_multi_plot(np.arange(num_time)+1, [eta_orig]+eta_correct, ['Original']+['Corrected ('+e+')' for e in correction_types], ['blue']+correction_colours, title='Average sea surface height (emulated)', units='m', dates=False)
            
            
    

