"""Utility functions for multi-GPU training with JAX."""

import jax


def replicate_to_devices(value, devices=None):
    """Replicate a value across specified devices."""
    if devices is None:
        devices = jax.local_devices()
    return jax.device_put_replicated(value, devices)


def pmean_gradient(grad, axis_name='i'):
    """Average gradients across devices."""
    return jax.lax.pmean(grad, axis_name=axis_name)


def create_synced_grad_fn(loss_fn, has_aux=False, axis_name='i'):
    """Create a gradient function that synchronizes gradients across devices."""
    grad_fn = jax.value_and_grad(loss_fn, has_aux=has_aux)
    
    def synced_fn(*args, **kwargs):
        if has_aux:
            (loss, aux), grad = grad_fn(*args, **kwargs)
            return (loss, aux), pmean_gradient(grad, axis_name)
        else:
            loss, grad = grad_fn(*args, **kwargs)
            return loss, pmean_gradient(grad, axis_name)
    
    return synced_fn