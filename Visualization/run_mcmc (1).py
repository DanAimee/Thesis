#I am struggling to understand this particular notebook specifically the argument parsers and am running into a ton of errors when I try and replicate this


import matplotlib

matplotlib.use("Agg")

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist

from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from catalax import Model, visualize
from catalax.mcmc import plot_corner, plot_trace, priors
from catalax.neural import NeuralODE

###########################
###### Example Usage ######
###########################

# python run_mcmc.py --model model_07.json  --parallel false --chains 1 --nsamples 10000 --warmup 10000

############################
##### Argument parsing #####
############################

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-c", "--chains", default=20, type=int)
parser.add_argument("-p", "--parallel", default=True, type=bool)

parser.add_argument("--nsamples", type=int, default=10_000)
parser.add_argument("--warmup", type=int, default=10_000)
parser.add_argument("--upper", type=float, default=1e4)
parser.add_argument("--lower", type=float, default=1e-6)

parser.add_argument("--dir", type=str, default="./results")

model_path = parser.parse_args().model
model_name = os.path.basename(model_path).split(".")[0]
dirpath = os.path.join(parser.parse_args().dir, model_name)
os.makedirs(dirpath, exist_ok=True)

PARALLEL = parser.parse_args().parallel
NUM_WORKERS = parser.parse_args().chains

if PARALLEL:
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(NUM_WORKERS)

#####################################
##### Pre-liminary data loading #####
#####################################

# Load trained neural ODE
neural_ode = NeuralODE.from_eqx("./assets/trained_model.eqx")

# Load the data
y0s = jnp.load("./assets/y0s.npy")
time = jnp.load("./assets/times.npy")
data = jnp.load("./assets/data.npy")
dataset_size, length_size, _ = data.shape

# Predict the trajectories of the given y0s
preds = jax.vmap(neural_ode, in_axes=(0, 0))(time, y0s)
dataset_size, length_size, _ = preds.shape
ins = preds.reshape(dataset_size * length_size, -1)

# Predict the rates of the given y0s
rates = jax.vmap(neural_ode.func, in_axes=(None, 0, None))(0.0, ins, 0.0)

# Load the catalax model
model = Model.load(model_path)
rate_func = model._setup_rate_function(in_axes=(0, 0, None))

####################
##### Run sHMC #####
####################

# Please note that the following surrogate steps are
# yet to be implemented in Catalax and thus are a bit
# techy within this script.

distributions = []
for name in model._get_parameter_order():
    parameter = model.parameters[name]
    parameter.prior = priors.Uniform(
        low=parser.parse_args().lower,
        high=parser.parse_args().upper,
    )

    # Manual cases (Need to be put into model JSON)
    if parameter.name == "kd":
        parameter.prior = priors.Uniform(low=1e-6, high=0.002)
    elif parameter.name == "k_d":
        parameter.prior = priors.Uniform(low=1e-6, high=0.002)
    elif "closure" in parameter.name.lower():
        parameter.prior = priors.Uniform(low=1e-8, high=1.0)
    elif parameter.name in ["k7", "k8"]:
        parameter.prior = priors.Uniform(low=1e-10, high=1.0)

    distributions.append((parameter.name, parameter.prior._distribution_fun))


def mcmc_model(times, ys, true_rates, yerrs):
    """The surrogate MCMC Model"""

    theta = jnp.array(
        [numpyro.sample(name, distribution) for name, distribution in distributions]
    )

    predicted_rates = rate_func(times, ys, theta)

    sigma = numpyro.sample("sigma", dist.Normal(0, yerrs))  # type: ignore

    numpyro.sample("rates", dist.Normal(predicted_rates, sigma), obs=true_rates)  # type: ignore


mcmc = MCMC(
    NUTS(mcmc_model, dense_mass=True),
    num_warmup=parser.parse_args().warmup,
    num_samples=parser.parse_args().nsamples,
    progress_bar=True,
    num_chains=1,  # NUM_WORKERS if PARALLEL else 1,
    chain_method="parallel" if PARALLEL else "sequential",
)

mcmc.run(
    PRNGKey(100),
    times=time.ravel(),
    ys=ins,
    true_rates=rates,
    yerrs=1e-4,
)

mcmc.print_summary()

#########################
##### MCMC Plotting #####
#########################

# Trace plot
f = plot_trace(mcmc, model)
plt.savefig(os.path.join(dirpath, f"{model_name}_trace.png"))

# Corner plot
plt.clf()
f = plot_corner(mcmc)
plt.savefig(os.path.join(dirpath, f"{model_name}_corner.png"))

########################
##### Model Saving #####
########################

# Add inferred mean parameters to the model
for parameter, samples in mcmc.get_samples().items():
    if parameter not in model.parameters:
        continue

    model.parameters[parameter].value = float(jnp.mean(samples))

# Save the model
model.save(os.path.join(dirpath), name=f"{model.name}_mcmc")

########################
##### Fit Plotting #####
########################

# Get initial conditions for plotting
initial_conditions = [
    {
        str(species): float(y0s[dataset, i])
        for i, species in enumerate(model._get_species_order())
    }
    for dataset in range(y0s.shape[0])
]

# Calculate AIC
aic = model.calculate_aic(data=data, initial_conditions=initial_conditions, times=time)
print(f"AIC value: {aic}")

# Plot the fit
f = visualize(
    model=model,
    data=data,
    times=time,
    initial_conditions=initial_conditions,
    figsize=(5, 8),
    mcmc=mcmc,
    save=os.path.join(dirpath, f"{model_name}_fit.png"),
    title="Test",
)
