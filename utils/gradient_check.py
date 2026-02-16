"""
Module for checking gradients to ensure they are correct or at least approximate.
"""

import numpy as np
import random


# Define functions for both analytical and numerical gradients to check
def numerical_gradient(layer, x, h=1e-5):
    """
    Function for numerical gradient (a.k.a direct calculus derivative approach)

    Formula -> f'(x) = lim  [f(x+h) - f(x-h)] / 2h
                       h→0
    """
    
    # Initialize the parameters
    output = layer.forward(x)
    loss = np.sum(output)
    dout = np.ones_like(output)  # Loss gradient with respect to output
    layer.backward(dout)
    params = layer.get_params()  # {"W": {"value":..., "grad":...}, ...}
    numerical_grads = dict()
    sample_dict = dict()

    # Iterate trough parameters
    for key in params.keys():
        # Take the parameter value and create an empty matrix for gradient
        param_value = params[key]["value"]
        grad = np.zeros(param_value.shape)

        # Take all index values in parameter values and sample them to achieve faster calculations
        all_indices = list(np.ndindex(param_value.shape))
        sampled = random.sample(all_indices, min(30, len(all_indices)))

        # Store the sampled index inside of the dict
        sample_dict[key] = sampled
        
        # Iterate trough parameter values via index
        for idx in sampled:
            # Original value
            original = param_value[idx]
            
            # Increasing the h
            param_value[idx] = original + h
            layer.set_params({key: param_value})
            loss_plus = np.sum(layer.forward(x))
            
            # Decreasing the h
            param_value[idx] = original - h
            layer.set_params({key: param_value})
            loss_minus = np.sum(layer.forward(x))
            
            # Set back to original param value
            param_value[idx] = original
            layer.set_params({key: param_value})
            
            # Apply formula
            grad[idx] = (loss_plus - loss_minus) / (2 * h)
        
        numerical_grads[key] = grad

    return numerical_grads, sample_dict

def gradient_check(layer, x):
    """
    Function to check gradients both analytically and numerically and measure the error between them.
    """

    # Take the numerical gradients and analytical gradients
    numerical_grads, sample_dict = numerical_gradient(layer, x)
    params = layer.get_params()
    errors = list()

    # Iterate trough parameters
    for key in params:
        # Create list for key errors
        key_errors = list()

        for idx in sample_dict[key]:
            # Take analyticial and numerical gradient values
            analytical_grad = params[key]["grad"][idx]
            numerical_grad = numerical_grads[key][idx]

            # Calculate relative error to compare grads
            rel_err = np.abs(analytical_grad - numerical_grad) / (np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8)  # 1e-8 is for preventing 0 denominator error
            key_errors.append(rel_err)

        # Take the maximum error
        max_error = np.max(key_errors)

        # Check the result based on error
        if max_error < 1e-7:
            result = "PASSED"
        elif max_error < 1e-5:
            result = "ACCEPTABLE"
        else:
            result = "FAILED"

        # Log the result
        print(f"{key}: max_error={max_error:.2e} → {result}")

        # Store the key error in general list
        errors.extend(key_errors)

    return np.max(errors)
